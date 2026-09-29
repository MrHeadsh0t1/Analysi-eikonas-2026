import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

import puzzle
import solver
from evaluation import neighbor_pairs


MODES = ("edge", "combined", "deep_only", "combined_deep", "full")
RESULTS_DIR = Path("results")


def measure(
    pieces: list[puzzle.Piece],
    placement: list[tuple[int, int]],
    rows: int,
    cols: int,
) -> dict:
    count = len(pieces)
    truth_grid = [0] * count

    for index, piece in enumerate(pieces):
        position = piece.original_row * cols + piece.original_col
        truth_grid[position] = index

    solved_grid = [index for index, turns in placement]
    true_neighbors = neighbor_pairs(truth_grid, rows, cols)

    absolute_position = sum(
        a == b for a, b in zip(solved_grid, truth_grid)
    )
    absolute_rotation = sum(
        turns == (-pieces[index].applied_rotation) % 4
        for index, turns in placement
    )
    absolute_neighbors = len(
        true_neighbors & neighbor_pairs(solved_grid, rows, cols)
    )

    grid_positions = np.arange(count).reshape(rows, cols)
    best = None

    # Η περιστροφή 90° αλλάζει τις διαστάσεις ορθογώνιου πλέγματος.
    allowed_turns = (0, 1, 2, 3) if rows == cols else (0, 2)

    for ccw_turns in allowed_turns:
        order = np.rot90(grid_positions, k=ccw_turns).ravel()
        aligned = [placement[int(i)] for i in order]
        aligned_grid = [index for index, turns in aligned]

        positions = sum(
            a == b for a, b in zip(aligned_grid, truth_grid)
        )
        rotations = sum(
            (turns - ccw_turns) % 4
            == (-pieces[index].applied_rotation) % 4
            for index, turns in aligned
        )
        neighbors = len(
            true_neighbors & neighbor_pairs(aligned_grid, rows, cols)
        )

        candidate = (positions, rotations, neighbors, ccw_turns)

        if best is None or (
            candidate[0] + candidate[1],
            candidate[2],
            candidate[0],
        ) > (
            best[0] + best[1],
            best[2],
            best[0],
        ):
            best = candidate

    return {
        "absolute_position": absolute_position,
        "absolute_neighbors": absolute_neighbors,
        "absolute_rotation": absolute_rotation,
        "aligned_position": best[0],
        "aligned_rotation": best[1],
        "aligned_neighbors": best[2],
        "alignment_degrees_ccw": best[3] * 90,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    rows = args.rows
    cols = args.cols

    if rows < 2 or cols < 2:
        parser.error(
            "Οι γραμμές και οι στήλες πρέπει να είναι τουλάχιστον 2."
        )

    images_dir = Path("test_images")
    if not images_dir.exists():
        raise ValueError("Δεν βρέθηκε ο φάκελος test_images.")

    images = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError("Βάλε εικόνες στον φάκελο test_images.")

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "experiment_results.csv"

    fields = [
        "image",
        "seed",
        "rows",
        "cols",
        "mode",
        "cost",
        "seconds",
        "absolute_position",
        "absolute_neighbors",
        "absolute_rotation",
        "aligned_position",
        "aligned_neighbors",
        "aligned_rotation",
        "alignment_degrees_ccw",
        "exact_pixels",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for image_path in images:
            for seed in args.seeds:
                original, pieces = puzzle.generate_puzzle(
                    str(image_path), rows, cols, seed
                )

                arrays = [
                    np.asarray(piece.image, dtype=np.float32) / 255.0
                    for piece in pieces
                ]

                prefix = f"{image_path.stem}_{rows}x{cols}_seed{seed}"

                original.save(
                    RESULTS_DIR / f"{prefix}_original.png"
                )
                puzzle.make_preview(pieces, rows, cols).save(
                    RESULTS_DIR / f"{prefix}_shuffled.png"
                )

                for mode in MODES:
                    solver.FEATURE_MODE = mode

                    print(
                        f"\nΕικόνα: {image_path.name} | "
                        f"πλέγμα: {rows}×{cols} | "
                        f"seed: {seed} | μέθοδος: {mode}",
                        flush=True,
                    )

                    start = time.perf_counter()
                    placement, cost = solver.solve(
                        arrays, rows, cols
                    )
                    seconds = time.perf_counter() - start

                    output_path = (
                        RESULTS_DIR / f"{prefix}_{mode}.png"
                    )
                    solver.save_reconstruction(
                        arrays,
                        placement,
                        str(output_path),
                        rows,
                        cols,
                    )

                    with Image.open(output_path) as result:
                        exact = np.array_equal(
                            np.asarray(original.convert("RGB")),
                            np.asarray(result.convert("RGB")),
                        )

                    row = {
                        "image": image_path.name,
                        "seed": seed,
                        "rows": rows,
                        "cols": cols,
                        "mode": mode,
                        "cost": round(cost, 6),
                        "seconds": round(seconds, 2),
                        **measure(pieces, placement, rows, cols),
                        "exact_pixels": exact,
                    }

                    writer.writerow(row)
                    file.flush()

                    neighbor_count = (
                        rows * (cols - 1)
                        + cols * (rows - 1)
                    )

                    print(
                        f"Θέση "
                        f"{row['absolute_position']}/{rows * cols}, "
                        f"γείτονες "
                        f"{row['absolute_neighbors']}/{neighbor_count}, "
                        f"περιστροφή "
                        f"{row['absolute_rotation']}/{rows * cols} | "
                        f"ευθυγραμμισμένη θέση "
                        f"{row['aligned_position']}/{rows * cols}"
                    )

    print(f"\nΑποτελέσματα: {csv_path}")


if __name__ == "__main__":
    main()