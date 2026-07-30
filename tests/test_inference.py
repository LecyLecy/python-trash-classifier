from __future__ import annotations

import unittest

import torch
from PIL import Image

from inference.api import PredictConfig, predict_pil_ui


LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("_logits", torch.tensor([logits], dtype=torch.float32))

    def forward(self, _image: torch.Tensor) -> torch.Tensor:
        return self._logits


class PredictionDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (320, 240), color=(225, 225, 225))

    def test_high_confidence_prediction_does_not_require_review(self) -> None:
        model = FixedLogitModel([8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = predict_pil_ui(
            self.image,
            model,
            LABELS,
            "cpu",
            PredictConfig(topk=3),
        )

        self.assertEqual(result["label"], "cardboard")
        self.assertGreater(result["confidence"], 0.99)
        self.assertFalse(result["needs_review"])
        self.assertEqual(len(result["top"]), 3)
        self.assertIn("cardboard recycling", result["instruction"])

    def test_glass_and_plastic_pair_always_requests_review(self) -> None:
        model = FixedLogitModel([0.0, 4.0, 0.0, 0.0, 3.0, 0.0])
        result = predict_pil_ui(
            self.image,
            model,
            LABELS,
            "cpu",
            PredictConfig(
                confidence_threshold=0.10,
                margin_threshold=0.01,
                topk=3,
            ),
        )

        self.assertEqual(result["label"], "glass")
        self.assertEqual(result["top"][1]["label"], "plastic")
        self.assertTrue(result["needs_review"])

    def test_close_predictions_request_review(self) -> None:
        model = FixedLogitModel([2.0, 0.0, 0.0, 1.9, 0.0, 0.0])
        result = predict_pil_ui(
            self.image,
            model,
            LABELS,
            "cpu",
            PredictConfig(
                confidence_threshold=0.10,
                margin_threshold=0.15,
                topk=2,
            ),
        )

        self.assertTrue(result["needs_review"])
        self.assertLess(result["margin"], 0.15)


if __name__ == "__main__":
    unittest.main()
