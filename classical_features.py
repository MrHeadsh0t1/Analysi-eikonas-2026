import numpy as np


STRIP_WIDTH = 8
COLOR_BINS = 8


def side_strip(image: np.ndarray, side: str) -> np.ndarray:
    """Επιστρέφει λωρίδα 8 pixels από την επιλεγμένη πλευρά."""
    if side == "top":
        return image[:STRIP_WIDTH, :, :]
    if side == "right":
        return image[:, -STRIP_WIDTH:, :]
    if side == "bottom":
        return image[-STRIP_WIDTH:, :, :]
    if side == "left":
        return image[:, :STRIP_WIDTH, :]

    raise ValueError(f"Άγνωστη πλευρά: {side}")


def color_histogram(strip: np.ndarray) -> np.ndarray:
    """Κανονικοποιημένο ιστόγραμμα RGB: 8 bins × 3 κανάλια."""
    histograms = []

    for channel in range(3):
        histogram, _ = np.histogram(
            strip[:, :, channel],
            bins=COLOR_BINS,
            range=(0.0, 1.0),
        )
        histogram = histogram.astype(np.float32)
        histogram /= max(histogram.sum(), 1.0)
        histograms.append(histogram)

    return np.concatenate(histograms)


def texture_descriptor(image: np.ndarray, side: str) -> np.ndarray:
    """Μέση ένταση και ενέργεια οριζόντιων/κάθετων μεταβολών."""
    grayscale = np.mean(image, axis=2)
    gradient_y, gradient_x = np.gradient(grayscale)

    gx = side_strip(gradient_x[:, :, None], side)[:, :, 0]
    gy = side_strip(gradient_y[:, :, None], side)[:, :, 0]

    return np.array(
        [
            np.mean(np.abs(gx)),
            np.mean(np.abs(gy)),
            np.mean(gx ** 2),
            np.mean(gy ** 2),
        ],
        dtype=np.float32,
    )


def describe_side(
    image: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        color_histogram(side_strip(image, side)),
        texture_descriptor(image, side),
    )


def descriptor_distances(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float]:
    first_color, first_texture = first
    second_color, second_texture = second

    color_distance = float(np.mean(np.abs(
        first_color - second_color
    )))
    texture_distance = float(np.mean(np.abs(
        first_texture - second_texture
    )))

    return color_distance, texture_distance