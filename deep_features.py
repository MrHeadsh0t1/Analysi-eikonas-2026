import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


SIDE_WIDTH = 32


class DeepFeatureExtractor:
    def __init__(self) -> None:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.eval()

        # Χρησιμοποιούμε ενδιάμεσους χάρτες χαρακτηριστικών
        # μέχρι και το layer2, όχι την τελική ταξινόμηση.
        self.backbone = nn.Sequential(*list(model.children())[:6])
        self.backbone.eval()
        self.preprocess = weights.transforms()

    def _descriptor(self, region: np.ndarray) -> np.ndarray:
        pixels = np.clip(region * 255, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels)
        tensor = self.preprocess(image).unsqueeze(0)

        with torch.inference_mode():
            feature_map = self.backbone(tensor)
            pooled = feature_map.mean(dim=(2, 3))
            vector = pooled[0].cpu().numpy()

        norm = np.linalg.norm(vector)
        return vector / max(norm, 1e-12)

    def describe_tile(self, tile: np.ndarray) -> np.ndarray:
        return self._descriptor(tile)

    def describe_many(
        self,
        regions: list[np.ndarray],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Εξάγει CNN περιγραφείς για πολλές περιοχές μαζί."""
        results = []

        for start in range(0, len(regions), batch_size):
            batch_regions = regions[start:start + batch_size]
            tensors = []

            for region in batch_regions:
                pixels = np.clip(region * 255, 0, 255).astype(np.uint8)
                image = Image.fromarray(pixels)
                tensors.append(self.preprocess(image))

            batch = torch.stack(tensors)

            with torch.inference_mode():
                maps = self.backbone(batch)
                vectors = maps.mean(dim=(2, 3))
                vectors = torch.nn.functional.normalize(
                    vectors, p=2, dim=1
                )

            results.append(vectors.cpu().numpy())

        return np.concatenate(results, axis=0)

    def describe_side(self, tile: np.ndarray, side: str) -> np.ndarray:
        if side == "top":
            region = tile[:SIDE_WIDTH, :, :]
        elif side == "right":
            region = tile[:, -SIDE_WIDTH:, :]
        elif side == "bottom":
            region = tile[-SIDE_WIDTH:, :, :]
        elif side == "left":
            region = tile[:, :SIDE_WIDTH, :]
        else:
            raise ValueError(f"Άγνωστη πλευρά: {side}")

        return self._descriptor(region)


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(first, second), -1.0, 1.0))