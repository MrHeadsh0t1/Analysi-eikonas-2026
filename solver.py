from pathlib import Path

import numpy as np
from PIL import Image
import json
from classical_features import describe_side, descriptor_distances
from deep_features import DeepFeatureExtractor, cosine_distance, SIDE_WIDTH
import argparse
from local_features import local_descriptor, local_distance

from features import border_distance, load_piece, rotate_clockwise

BEAM_WIDTH = 2000
FEATURE_MODE = "full"  # "edge", "combined", "combined_deep", "full"
LOCAL_WEIGHT = 0.002
COLOR_WEIGHT = 0.02
TEXTURE_WEIGHT = 0.2
DEEP_SIDE_WEIGHT = 0.01
DEEP_TILE_WEIGHT = 0.005


def vertical_distance(top_piece: np.ndarray, bottom_piece: np.ndarray) -> float:
    # Με ανταλλαγή των αξόνων, το κάτω/πάνω όριο γίνεται δεξί/αριστερό.
    top = np.transpose(top_piece, (1, 0, 2))
    bottom = np.transpose(bottom_piece, (1, 0, 2))
    return border_distance(top, bottom)


def solve(
    pieces: list[np.ndarray],
    rows: int = 3,
    cols: int = 3,
) -> tuple[list[tuple[int, int]], float]:
    count = len(pieces)
    if rows < 2 or cols < 2:
        raise ValueError("Οι γραμμές και οι στήλες πρέπει να είναι τουλάχιστον 2.")
    if count != rows * cols:
        raise ValueError(f"Αναμένονται {rows * cols} κομμάτια, βρέθηκαν {count}.")

    # Κάθε κατάσταση είναι (δείκτης κομματιού, περιστροφές δεξιόστροφα).
    states = [(index, turns) for index in range(count) for turns in range(4)]
    rotated = {
        (index, turns): rotate_clockwise(pieces[index], turns)
        for index, turns in states
    }

    descriptors = {
        state: {
            side: describe_side(image, side)
            for side in ("top", "right", "bottom", "left")
        }
        for state, image in rotated.items()
    }

    local_descriptors = {}

    if FEATURE_MODE == "full":
        print("Εξαγωγή τοπικών χαρακτηριστικών Harris...")
        local_descriptors = {
            state: {
                side: local_descriptor(image, side)
                for side in ("top", "right", "bottom", "left")
            }
            for state, image in rotated.items()
        }

    deep_descriptors = {}

    if FEATURE_MODE in ("deep_only", "combined_deep", "full"):
        print("Εξαγωγή CNN χαρακτηριστικών...")
        extractor = DeepFeatureExtractor()

        deep_keys = [
            (state, side)
            for state in states
            for side in ("top", "right", "bottom", "left", "tile")
        ]

        regions = []
        for state, side in deep_keys:
            image = rotated[state]

            if side == "top":
                region = image[:SIDE_WIDTH, :, :]
            elif side == "right":
                region = image[:, -SIDE_WIDTH:, :]
            elif side == "bottom":
                region = image[-SIDE_WIDTH:, :, :]
            elif side == "left":
                region = image[:, :SIDE_WIDTH, :]
            else:
                region = image

            regions.append(region)

        vectors = extractor.describe_many(regions)
        deep_descriptors = dict(zip(deep_keys, vectors))
        print(f"Υπολογίστηκαν {len(deep_descriptors)} CNN περιγραφείς")

    white_fraction = float(np.mean([
        np.mean(np.all(piece > 0.95, axis=2))
        for piece in pieces
    ]))

    white_background_weight = (
        0.1
        if white_fraction > 0.30 and FEATURE_MODE != "deep_only"
        else 0.0
    )
    print(f"Ποσοστό λευκού φόντου: {white_fraction:.1%}")

    outer_cost = {}
    for state, image in rotated.items():
        borders = (
            image[0, :, :],    # πάνω
            image[:, -1, :],   # δεξιά
            image[-1, :, :],   # κάτω
            image[:, 0, :],    # αριστερά
        )
        outer_cost[state] = tuple(
            float(np.mean((border - 1.0) ** 2))
            for border in borders
        )

    horizontal = {}
    vertical = {}

    for first in states:
        for second in states:
            if first[0] == second[0]:
                continue

            horizontal[first, second] = (
                0.0
                if FEATURE_MODE == "deep_only"
                else border_distance(rotated[first], rotated[second])
            )
            vertical[first, second] = (
                0.0
                if FEATURE_MODE == "deep_only"
                else vertical_distance(rotated[first], rotated[second])
            )
            if FEATURE_MODE in ("combined", "combined_deep", "full"):
                h_color, h_texture = descriptor_distances(
                    descriptors[first]["right"],
                    descriptors[second]["left"],
                )
                v_color, v_texture = descriptor_distances(
                    descriptors[first]["bottom"],
                    descriptors[second]["top"],
                )

                horizontal[first, second] += (
                    COLOR_WEIGHT * h_color
                    + TEXTURE_WEIGHT * h_texture
                )
                vertical[first, second] += (
                    COLOR_WEIGHT * v_color
                    + TEXTURE_WEIGHT * v_texture
                )
            if FEATURE_MODE in ("combined_deep", "deep_only", "full"):
                tile_distance = cosine_distance(
                    deep_descriptors[first, "tile"],
                    deep_descriptors[second, "tile"],
                )
                h_deep = cosine_distance(
                    deep_descriptors[first, "right"],
                    deep_descriptors[second, "left"],
                )
                v_deep = cosine_distance(
                    deep_descriptors[first, "bottom"],
                    deep_descriptors[second, "top"],
                )

                horizontal[first, second] += (
                    DEEP_SIDE_WEIGHT * h_deep
                    + DEEP_TILE_WEIGHT * tile_distance
                )
                vertical[first, second] += (
                    DEEP_SIDE_WEIGHT * v_deep
                    + DEEP_TILE_WEIGHT * tile_distance
                )
            if FEATURE_MODE == "full":
                h_local = local_distance(
                    local_descriptors[first]["right"],
                    local_descriptors[second]["left"],
                )
                v_local = local_distance(
                    local_descriptors[first]["bottom"],
                    local_descriptors[second]["top"],
                )

                horizontal[first, second] += LOCAL_WEIGHT * h_local
                vertical[first, second] += LOCAL_WEIGHT * v_local

    # (συνολικό κόστος, τοποθετημένες καταστάσεις, χρησιμοποιημένα κομμάτια)
    beam = [(0.0, (), frozenset())]

    for position in range(count):
        row, col = divmod(position, cols)
        candidates = []

        for cost, placement, used in beam:
            for state in states:
                piece_index = state[0]
                if piece_index in used:
                    continue

                new_cost = cost

                if col > 0:
                    new_cost += horizontal[placement[-1], state]

                if row > 0:
                    above = placement[position - cols]
                    new_cost += vertical[above, state]

                if white_background_weight > 0:
                    top, right, bottom, left = outer_cost[state]
                    boundary_cost = (
                        (top if row == 0 else 0.0)
                        + (right if col == cols - 1 else 0.0)
                        + (bottom if row == rows - 1 else 0.0)
                        + (left if col == 0 else 0.0)
                    )
                    new_cost += white_background_weight * boundary_cost
                candidates.append((
                    new_cost,
                    placement + (state,),
                    used | {piece_index},
                ))

        candidates.sort(key=lambda item: item[0])
        beam = candidates[:BEAM_WIDTH]
        print(f"Τοποθετήθηκαν {position + 1}/{count} κομμάτια")

    best_cost, best_placement, _ = beam[0]
    return list(best_placement), best_cost


def save_reconstruction(
    pieces: list[np.ndarray],
    placement: list[tuple[int, int]],
    output_path: str,
    rows: int = 3,
    cols: int = 3,
) -> None:
    tile_height, tile_width = pieces[0].shape[:2]

    canvas = Image.new(
        "RGB",
        (cols * tile_width, rows * tile_height),
    )

    for position, (index, turns) in enumerate(placement):
        row, col = divmod(position, cols)
        rotated = rotate_clockwise(pieces[index], turns)
        pixels = np.clip(rotated * 255, 0, 255).astype(np.uint8)

        canvas.paste(
            Image.fromarray(pixels),
            (col * tile_width, row * tile_height),
        )

    canvas.save(output_path)


def main() -> None:
    global FEATURE_MODE

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("edge", "combined", "deep_only", "combined_deep", "full"),
        default=FEATURE_MODE,
    )
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    args = parser.parse_args()
    FEATURE_MODE = args.mode
    config = json.loads(
    Path("output/puzzle_config.json").read_text(encoding="utf-8")
    )
    rows = config["rows"] if args.rows is None else args.rows
    cols = config["cols"] if args.cols is None else args.cols

    print(f"Μέθοδος: {FEATURE_MODE}")

    paths = sorted(Path("output/pieces").glob("piece_*.png"))
    pieces = [load_piece(path) for path in paths]

    placement, cost = solve(pieces, rows, cols)
    save_reconstruction(
    pieces, placement, "output/reconstructed.png", rows, cols
    )

    Path("output/solution.json").write_text(
        json.dumps(
            [
                {"piece": index, "turns": turns}
                for index, turns in placement
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nΣυνολικό κόστος: {cost:.6f}")
    print("Πλέγμα (κομμάτι, δεξιόστροφη διόρθωση):")

    for row in range(rows):
        cells = placement[row * cols:(row + 1) * cols]
        print(" | ".join(
            f"{index:02d} / {turns * 90:3d}°"
            for index, turns in cells
        ))

    print("\nΑποθηκεύτηκε το output/reconstructed.png")


if __name__ == "__main__":
    main()