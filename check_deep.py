from pathlib import Path

from deep_features import DeepFeatureExtractor, cosine_distance
from features import load_piece


paths = sorted(Path("output/pieces").glob("piece_*.png"))
if len(paths) != 9:
    raise ValueError("Δεν βρέθηκαν 9 κομμάτια.")

pieces = [load_piece(path) for path in paths]
extractor = DeepFeatureExtractor()

tile_vector = extractor.describe_tile(pieces[0])
right_vector = extractor.describe_side(pieces[0], "right")
left_vector = extractor.describe_side(pieces[3], "left")

print("Διάσταση περιγραφέα κομματιού:", tile_vector.shape)
print("Διάσταση περιγραφέα πλευράς:", right_vector.shape)
print(
    "CNN απόσταση πλευρών piece_00 → piece_03:",
    f"{cosine_distance(right_vector, left_vector):.6f}",
)