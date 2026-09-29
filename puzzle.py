import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


GRID_ROWS = 3
GRID_COLS = 3
TILE_SIZE = 128
SEED = 42


@dataclass
class Piece:
    image: Image.Image
    original_row: int
    original_col: int
    applied_rotation: int  # Δεξιόστροφες περιστροφές των 90°


def generate_puzzle(
    image_path: str,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    seed: int | None = None,
) -> tuple[Image.Image, list[Piece]]:
    if rows < 2 or cols < 2:
        raise ValueError("Οι γραμμές και οι στήλες πρέπει να είναι τουλάχιστον 2.")

    size = (cols * TILE_SIZE, rows * TILE_SIZE)

    with Image.open(image_path) as source:
        original = ImageOps.fit(source.convert("RGB"), size)

    pieces = []

    for row in range(rows):
        for col in range(cols):
            x = col * TILE_SIZE
            y = row * TILE_SIZE
            tile = original.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
            pieces.append(Piece(tile, row, col, 0))

    rng = random.Random(SEED if seed is None else seed)
    rng.shuffle(pieces)

    for piece in pieces:
        piece.applied_rotation = rng.randrange(4)
        piece.image = piece.image.rotate(-90 * piece.applied_rotation)

    return original, pieces


def make_preview(
    pieces: list[Piece],
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> Image.Image:
    if len(pieces) != rows * cols:
        raise ValueError("Ο αριθμός κομματιών δεν ταιριάζει στο πλέγμα.")

    preview = Image.new(
        "RGB",
        (cols * TILE_SIZE, rows * TILE_SIZE),
    )

    for index, piece in enumerate(pieces):
        row, col = divmod(index, cols)
        preview.paste(
            piece.image,
            (col * TILE_SIZE, row * TILE_SIZE),
        )

    return preview


def save_puzzle(
    original: Image.Image,
    pieces: list[Piece],
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> None:
    if len(pieces) != rows * cols:
        raise ValueError("Ο αριθμός κομματιών δεν ταιριάζει στο πλέγμα.")

    output = Path("output")
    pieces_dir = output / "pieces"
    pieces_dir.mkdir(parents=True, exist_ok=True)

    # Αφαιρούμε κομμάτια προηγούμενου πλέγματος.
    for old_piece in pieces_dir.glob("piece_*.png"):
        old_piece.unlink()

    # Η προηγούμενη λύση δεν ισχύει για το νέο παζλ.
    for old_result in ("solution.json", "reconstructed.png"):
        (output / old_result).unlink(missing_ok=True)

    original.save(output / "original.png")
    make_preview(pieces, rows, cols).save(output / "shuffled.png")

    for index, piece in enumerate(pieces):
        piece.image.save(pieces_dir / f"piece_{index:02d}.png")

    ground_truth = [
        {
            "piece": index,
            "row": piece.original_row,
            "col": piece.original_col,
            "correct_turns": (-piece.applied_rotation) % 4,
        }
        for index, piece in enumerate(pieces)
    ]

    (output / "puzzle_config.json").write_text(
        json.dumps(
            {"rows": rows, "cols": cols, "tile_size": TILE_SIZE},
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2),
        encoding="utf-8",
    )

    print("Αποθηκεύτηκαν τα κομμάτια στον φάκελο output/pieces/")
    print("Η σωστή λύση, μόνο για αξιολόγηση:")

    for index, piece in enumerate(pieces):
        correction = (-piece.applied_rotation) % 4
        print(
            f"piece_{index:02d}: θέση "
            f"({piece.original_row}, {piece.original_col}), "
            f"διόρθωση περιστροφής {correction * 90}° δεξιόστροφα"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Δημιουργία παζλ εικόνας")
    parser.add_argument("--image", default="input.jpg")
    parser.add_argument("--rows", type=int, default=GRID_ROWS)
    parser.add_argument("--cols", type=int, default=GRID_COLS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε η εικόνα: {image_path}")

    original, pieces = generate_puzzle(
        str(image_path),
        args.rows,
        args.cols,
        args.seed,
    )
    save_puzzle(original, pieces, args.rows, args.cols)