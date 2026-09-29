from pathlib import Path

import numpy as np
from PIL import Image


PIECES_DIR = Path("output/pieces")


def load_piece(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def rotate_clockwise(piece: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(piece, k=-turns)


def border_distance(left_piece: np.ndarray, right_piece: np.ndarray) -> float:
    # Συνέχεια χρώματος ακριβώς στη σύνδεση.
    seam = np.mean(
        (left_piece[:, -1, :] - right_piece[:, 0, :]) ** 2
    )

    # Πρόβλεψη της πρώτης στήλης του δεξιού κομματιού
    # από τις δύο τελευταίες στήλες του αριστερού.
    predicted_right = (
        2 * left_piece[:, -1, :] - left_piece[:, -2, :]
    )
    forward = np.mean(
        (predicted_right - right_piece[:, 0, :]) ** 2
    )

    # Αντίστοιχη πρόβλεψη προς την αντίθετη κατεύθυνση.
    predicted_left = (
        2 * right_piece[:, 0, :] - right_piece[:, 1, :]
    )
    backward = np.mean(
        (predicted_left - left_piece[:, -1, :]) ** 2
    )

    return float(seam + 0.5 * forward + 0.5 * backward)


def main() -> None:
    paths = sorted(PIECES_DIR.glob("piece_*.png"))

    if len(paths) != 9:
        raise ValueError(f"Αναμένονται 9 κομμάτια, βρέθηκαν {len(paths)}.")

    pieces = [load_piece(path) for path in paths]
    results = []

    reference_index = 0

    for reference_turns in range(4):
        rotated_reference = rotate_clockwise(
            pieces[reference_index], reference_turns
        )

        for candidate_index, candidate in enumerate(pieces):
            if candidate_index == reference_index:
                continue

            for candidate_turns in range(4):
                rotated_candidate = rotate_clockwise(
                    candidate, candidate_turns
                )

                distance = border_distance(
                    rotated_reference, rotated_candidate
                )

                results.append((
                    distance,
                    candidate_index,
                    reference_turns,
                    candidate_turns,
                ))

    results.sort()

    print("Οι 10 καλύτερες οριζόντιες συνδέσεις του piece_00:\n")

    for distance, candidate, ref_turns, candidate_turns in results[:10]:
        print(
            f"piece_00 ({ref_turns * 90:3d}°) → "
            f"piece_{candidate:02d} ({candidate_turns * 90:3d}°): "
            f"{distance:.6f}"
        )


if __name__ == "__main__":
    main()