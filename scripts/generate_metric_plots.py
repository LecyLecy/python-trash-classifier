"""Generate portfolio-ready diagnostic plots from project artifacts.

Run from the repository root after training and evaluation:

    python scripts/generate_metric_plots.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "trashnet" / "raw"
METRICS_DIR = ROOT / "metrics"
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
COLORS = ("#B7794B", "#45A99A", "#718096", "#D7A93E", "#4A86E8", "#7B6F83")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def save_figure(figure: plt.Figure, filename: str) -> None:
    figure.savefig(METRICS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_class_distribution() -> None:
    counts = {
        label: sum(
            path.suffix.lower() in IMAGE_EXTENSIONS
            for path in (DATA_DIR / label).iterdir()
        )
        for label in CLASSES
    }

    figure, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(
        [label.title() for label in CLASSES],
        counts.values(),
        color=COLORS,
        edgecolor="white",
        linewidth=1.2,
    )
    axis.bar_label(bars, padding=4, fontsize=9)
    axis.set_title("TrashNet class distribution", loc="left", fontsize=15, weight="bold")
    axis.set_ylabel("Images")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)
    figure.tight_layout()
    save_figure(figure, "class_distribution.png")


def plot_training_curves() -> None:
    log_path = METRICS_DIR / "train_log.csv"
    if not log_path.exists():
        return

    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_acc = [float(row["train_acc"]) for row in rows]
    val_acc = [float(row["val_acc"]) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    figure.suptitle("Training diagnostics", x=0.08, ha="left", fontsize=15, weight="bold")

    axes[0].plot(epochs, train_loss, marker="o", label="Train loss", color="#167D5A")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation loss", color="#D7A93E")
    axes[0].set_title("Loss", loc="left", fontsize=11, weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, train_acc, marker="o", label="Train accuracy", color="#167D5A")
    axes[1].plot(epochs, val_acc, marker="o", label="Validation accuracy", color="#D7A93E")
    axes[1].set_title("Accuracy", loc="left", fontsize=11, weight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)

    fine_tune_start = next(
        (int(row["epoch"]) for row in rows if row["stage"] != "classifier_head"),
        None,
    )
    if fine_tune_start is not None:
        for axis in axes:
            axis.axvline(
                fine_tune_start - 0.5,
                linestyle="--",
                color="#7B6F83",
                alpha=0.7,
            )
            axis.text(
                fine_tune_start - 0.35,
                0.96 if axis is axes[1] else max(train_loss) * 0.95,
                "Fine-tune layer4",
                fontsize=8,
                color="#7B6F83",
            )

    figure.tight_layout()
    save_figure(figure, "training_curves.png")


def plot_per_class_metrics() -> None:
    report_path = METRICS_DIR / "classification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    precision = [report[label]["precision"] for label in CLASSES]
    recall = [report[label]["recall"] for label in CLASSES]
    f1_score = [report[label]["f1-score"] for label in CLASSES]

    positions = list(range(len(CLASSES)))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar([p - width for p in positions], precision, width, label="Precision", color="#167D5A")
    axis.bar(positions, recall, width, label="Recall", color="#D7A93E")
    axis.bar([p + width for p in positions], f1_score, width, label="F1 score", color="#4A86E8")
    axis.set_xticks(positions, [label.title() for label in CLASSES])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title("Per-class test-set performance", loc="left", fontsize=15, weight="bold")
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)
    figure.tight_layout()
    save_figure(figure, "per_class_metrics.png")


def plot_sample_grid() -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10, 6.8))
    for axis, label in zip(axes.flat, CLASSES):
        candidates = sorted(
            path for path in (DATA_DIR / label).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        with Image.open(candidates[0]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(label.title(), weight="bold")
        axis.axis("off")
    figure.suptitle("Representative TrashNet samples", fontsize=15, weight="bold")
    figure.tight_layout()
    save_figure(figure, "sample_grid.png")


if __name__ == "__main__":
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_distribution()
    plot_training_curves()
    plot_per_class_metrics()
    plot_sample_grid()
    print("Generated metric plots in", METRICS_DIR)
