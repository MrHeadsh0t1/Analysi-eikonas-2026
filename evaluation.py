import json
from pathlib import Path

from PIL import Image
import numpy as np


OUTPUT = Path("output")


def neighbor_pairs(
    grid: list[int],
    rows: int = 3,
    cols: int = 3,
) -> set[frozenset[int]]:
    pairs = set()

    for row in range(rows):
        for col in range(cols):
            position = row * cols + col

            if col + 1 < cols:
                pairs.add(frozenset((grid[position], grid[position + 1])))

            if row + 1 < rows:
                pairs.add(frozenset((grid[position], grid[position + cols])))

    return pairs


def main() -> None:
    config = json.loads(
    (OUTPUT / "puzzle_config.json").read_text(encoding="utf-8")
    )
    rows = config["rows"]
    cols = config["cols"]

    truth = json.loads(
        (OUTPUT / "ground_truth.json").read_text(encoding="utf-8")
    )
    solution = json.loads(
        (OUTPUT / "solution.json").read_text(encoding="utf-8")
    )

    count = rows * cols
    if len(truth) != count or len(solution) != count:
        raise ValueError(f"Τα αρχεία πρέπει να περιέχουν {count} κομμάτια.")

    truth_by_piece = {item["piece"]: item for item in truth}
    solved_grid = [item["piece"] for item in solution]

    if set(solved_grid) != set(range(count)):
        raise ValueError("Η λύση δεν χρησιμοποιεί κάθε κομμάτι μία φορά.")

    correct_positions = 0
    correct_rotations = 0

    for position, item in enumerate(solution):
        piece = item["piece"]
        expected = truth_by_piece[piece]
        expected_position = (
            expected["row"] * cols + expected["col"]
        )

        if position == expected_position:
            correct_positions += 1

        if item["turns"] == expected["correct_turns"]:
            correct_rotations += 1

    truth_grid = [0] * count
    for piece, item in truth_by_piece.items():
        position = item["row"] * cols + item["col"]
        truth_grid[position] = piece

    true_neighbors = neighbor_pairs(truth_grid, rows, cols)
    solved_neighbors = neighbor_pairs(solved_grid, rows, cols)
    recovered_neighbors = len(true_neighbors & solved_neighbors)

    with Image.open(OUTPUT / "original.png") as original:
        original_pixels = np.asarray(original.convert("RGB"))
    with Image.open(OUTPUT / "reconstructed.png") as reconstructed:
        reconstructed_pixels = np.asarray(reconstructed.convert("RGB"))

    exact_match = np.array_equal(original_pixels, reconstructed_pixels)

    print(f"Ακρίβεια θέσης: {correct_positions}/{count} "
          f"= {correct_positions / count:.1%}")
    print(f"Ακρίβεια γειτόνων: "
          f"{recovered_neighbors}/{len(true_neighbors)} "
          f"= {recovered_neighbors / len(true_neighbors):.1%}")
    print(f"Ακρίβεια περιστροφής: {correct_rotations}/{count} "
          f"= {correct_rotations / count:.1%}")
    print(f"Ακριβής ταύτιση εικόνας: {'Ναι' if exact_match else 'Όχι'}")

    # Πρόσθετη αξιολόγηση έως συνολική περιστροφή του παζλ.
    best = None
    positions = np.arange(count).reshape(rows, cols)

    for ccw_turns in ((0, 1, 2, 3) if rows == cols else (0, 2)):
        rotated_positions = np.rot90(positions, k=ccw_turns).ravel()
        aligned_solution = [solution[int(i)] for i in rotated_positions]

        aligned_positions = sum(
            position == (
                truth_by_piece[item["piece"]]["row"] * cols
                + truth_by_piece[item["piece"]]["col"]
            )
            for position, item in enumerate(aligned_solution)
        )

        # Η συνολική αριστερόστροφη στροφή αλλάζει και τον
        # προσανατολισμό κάθε κομματιού.
        aligned_rotations = sum(
            (item["turns"] - ccw_turns) % 4
            == truth_by_piece[item["piece"]]["correct_turns"]
            for item in aligned_solution
        )

        aligned_grid = [item["piece"] for item in aligned_solution]
        aligned_neighbors = len(
            true_neighbors & neighbor_pairs(aligned_grid, rows, cols)
        )

        candidate = (
            aligned_positions,
            aligned_rotations,
            aligned_neighbors,
            ccw_turns,
        )
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

    aligned_positions, aligned_rotations, aligned_neighbors, turns = best

    print("\nΜετά από ευθυγράμμιση συνολικής περιστροφής:")
    print(f"Στροφή ευθυγράμμισης: {turns * 90}° αριστερόστροφα")
    print(f"Ακρίβεια θέσης: {aligned_positions}/{count}")
    print(f"Ακρίβεια γειτόνων: "
          f"{aligned_neighbors}/{len(true_neighbors)}")
    print(f"Ακρίβεια περιστροφής: {aligned_rotations}/{count}")


if __name__ == "__main__":
    main()