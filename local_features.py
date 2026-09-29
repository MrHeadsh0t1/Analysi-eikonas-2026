import numpy as np


BORDER_WIDTH = 16
PATCH_SIZE = 16
MAX_POINTS = 32


def box_sum_3x3(array: np.ndarray) -> np.ndarray:
    height, width = array.shape
    padded = np.pad(array, 1, mode="reflect")

    return sum(
        padded[dy:dy + height, dx:dx + width]
        for dy in range(3)
        for dx in range(3)
    )


def border_mask(shape: tuple[int, int], side: str) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)

    if side == "top":
        mask[:BORDER_WIDTH, :] = True
    elif side == "right":
        mask[:, -BORDER_WIDTH:] = True
    elif side == "bottom":
        mask[-BORDER_WIDTH:, :] = True
    elif side == "left":
        mask[:, :BORDER_WIDTH] = True
    else:
        raise ValueError(f"Άγνωστη πλευρά: {side}")

    return mask


def local_descriptor(image: np.ndarray, side: str) -> np.ndarray:
    """Μέσος περιγραφέας 128 τιμών από σημεία Harris μιας πλευράς."""
    grayscale = image.mean(axis=2)
    gy, gx = np.gradient(grayscale)

    # Πίνακας δομής και απόκριση Harris.
    sxx = box_sum_3x3(gx * gx)
    syy = box_sum_3x3(gy * gy)
    sxy = box_sum_3x3(gx * gy)

    response = (
        sxx * syy - sxy * sxy
        - 0.04 * (sxx + syy) ** 2
    )

    mask = border_mask(grayscale.shape, side)
    positive = response[mask & (response > 0)]

    if positive.size == 0:
        return np.zeros(128, dtype=np.float32)

    threshold = np.percentile(positive, 75)
    local_maximum = np.maximum.reduce([
        np.roll(np.roll(response, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ])

    points = np.argwhere(
        mask & (response >= threshold) & (response == local_maximum)
    )
    points = sorted(
        points,
        key=lambda point: response[point[0], point[1]],
        reverse=True,
    )[:MAX_POINTS]

    if not points:
        return np.zeros(128, dtype=np.float32)

    magnitude = np.hypot(gx, gy)
    orientation = (np.arctan2(gy, gx) + 2 * np.pi) % (2 * np.pi)
    bins = np.floor(orientation * 8 / (2 * np.pi)).astype(int)
    bins = np.clip(bins, 0, 7)

    pad = PATCH_SIZE // 2
    padded_magnitude = np.pad(magnitude, pad, mode="reflect")
    padded_bins = np.pad(bins, pad, mode="reflect")

    descriptors = []

    for y, x in points:
        patch_magnitude = padded_magnitude[
            y:y + PATCH_SIZE, x:x + PATCH_SIZE
        ]
        patch_bins = padded_bins[
            y:y + PATCH_SIZE, x:x + PATCH_SIZE
        ]

        histograms = []
        for cell_y in range(4):
            for cell_x in range(4):
                ys = slice(cell_y * 4, (cell_y + 1) * 4)
                xs = slice(cell_x * 4, (cell_x + 1) * 4)

                histogram = np.bincount(
                    patch_bins[ys, xs].ravel(),
                    weights=patch_magnitude[ys, xs].ravel(),
                    minlength=8,
                )
                histograms.append(histogram)

        vector = np.concatenate(histograms)
        vector /= max(np.linalg.norm(vector), 1e-12)
        descriptors.append(vector)

    result = np.mean(descriptors, axis=0)
    result /= max(np.linalg.norm(result), 1e-12)
    return result.astype(np.float32)


def local_distance(first: np.ndarray, second: np.ndarray) -> float:
    if not np.any(first) or not np.any(second):
        return 1.0

    return float(
        1.0 - np.clip(np.dot(first, second), -1.0, 1.0)
    )