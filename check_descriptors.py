from pathlib import Path

from classical_features import describe_side, descriptor_distances
from features import load_piece


pieces = [
    load_piece(path)
    for path in sorted(Path("output/pieces").glob("piece_*.png"))
]

if len(pieces) != 9:
    raise ValueError("Δεν βρέθηκαν 9 κομμάτια.")

right_00 = describe_side(pieces[0], "right")

for index in (1, 3, 7):
    left_candidate = describe_side(pieces[index], "left")
    color, texture = descriptor_distances(right_00, left_candidate)

    print(
        f"piece_00 → piece_{index:02d}: "
        f"χρώμα={color:.6f}, υφή={texture:.6f}"
    )