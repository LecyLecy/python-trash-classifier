from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .config import Config
from .dataset import Sample, TrashDataset, build_transforms
from .model import create_model
from .utils import set_seed, ensure_dir, save_json, get_device

def gather_samples(data_dir: Path, classes: Tuple[str, ...]) -> List[Sample]:
    samples: List[Sample] = []
    for i, cls in enumerate(classes):
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        for p in cls_dir.glob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                samples.append(Sample(path=p, label_idx=i))
    if not samples:
        raise RuntimeError("No images found. Check dataset path.")
    return samples

def split_samples(samples: List[Sample], seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    y = [s.label_idx for s in samples]
    train, temp = train_test_split(samples, test_size=(1.0-train_ratio), random_state=seed, stratify=y)
    y_temp = [s.label_idx for s in temp]
    val_size = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(temp, test_size=(1.0-val_size), random_state=seed, stratify=y_temp)
    return train, val, test

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    frozen_modules: Iterable[nn.Module] = (),
) -> Tuple[float, float]:
    model.train()
    for module in frozen_modules:
        module.eval()

    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


def set_trainable_modules(model: nn.Module, modules: Iterable[nn.Module]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]

def main():
    cfg = Config()
    set_seed(cfg.seed)

    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.metrics_dir)

    device = get_device(cfg.device)
    print(f"Using device: {device}")

    samples = gather_samples(cfg.data_dir, cfg.classes)
    train_s, val_s, _ = split_samples(
        samples,
        cfg.seed,
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
    )

    train_tf, eval_tf = build_transforms(cfg.image_size)
    train_ds = TrashDataset(train_s, transform=train_tf)
    val_ds = TrashDataset(val_s, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = create_model(
        cfg.model_name,
        num_classes=len(cfg.classes),
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    head_modules = (model.fc,)
    fine_tune_modules = (model.layer4, model.fc)
    frozen_for_head = (
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
    )
    frozen_for_fine_tuning = (
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
    )

    set_trainable_modules(model, head_modules)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    stage = "classifier_head"
    frozen_modules = frozen_for_head

    best_val_acc = -1.0
    best_path = cfg.output_dir / "model.pth"
    labels_path = cfg.output_dir / "labels.json"

    # save labels order once
    save_json(labels_path, {"classes": list(cfg.classes)})

    log_path = cfg.metrics_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "stage", "train_loss", "train_acc", "val_loss", "val_acc"])

    patience_left = cfg.patience

    for epoch in range(1, cfg.num_epochs + 1):
        if epoch == cfg.head_epochs + 1:
            model.load_state_dict(
                torch.load(best_path, map_location=device, weights_only=True)
            )
            set_trainable_modules(model, fine_tune_modules)
            optimizer = torch.optim.AdamW(
                trainable_parameters(model),
                lr=cfg.fine_tune_lr,
                weight_decay=cfg.weight_decay,
            )
            stage = "fine_tune_layer4"
            frozen_modules = frozen_for_fine_tuning
            patience_left = cfg.patience
            print(
                f"Fine-tuning layer4 + classifier at lr={cfg.fine_tune_lr:g}"
            )

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            frozen_modules,
        )
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    stage,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                ]
            )

        print(
            f"Epoch {epoch}/{cfg.num_epochs} [{stage}] | "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            patience_left = cfg.patience
            print(f"  Saved best -> {best_path} (val_acc={best_val_acc:.3f})")
        else:
            if stage == "fine_tune_layer4":
                patience_left -= 1
                if patience_left <= 0:
                    print("Early stopping.")
                    break

    # write summary
    save_json(cfg.metrics_dir / "metrics.json", {
        "best_val_acc": best_val_acc,
        "model_name": cfg.model_name,
        "image_size": cfg.image_size,
        "classes": list(cfg.classes),
        "training_strategy": "classifier_head_then_layer4",
    })

    print("Done. Next: run eval to produce confusion matrix if you want.")
    print("Tip: integrate using inference/predict.py")

if __name__ == "__main__":
    main()
