from io import BytesIO
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import streamlit as st

import puzzle
import solver
from evaluation import neighbor_pairs


st.set_page_config(page_title="Lost in Pieces", layout="wide")
st.title("Lost in Pieces — Ανακατασκευή εικόνας")
st.write(
    "Ανέβασε εικόνα, επίλεξε μέθοδο και δες πώς "
    "ανακατασκευάζεται ένα παζλ PxQ με περιστροφές."
)

uploaded = st.file_uploader(
    "Εικόνα εισόδου",
    type=["jpg", "jpeg", "png"],
)
mode = st.selectbox(
    "Μέθοδος",
    ("edge", "combined", "deep_only", "combined_deep", "full"),
)
grid_left, grid_right = st.columns(2)

rows = int(grid_left.number_input(
    "P — γραμμές", min_value=2, max_value=5, value=3
))
cols = int(grid_right.number_input(
    "Q — στήλες", min_value=2, max_value=5, value=3
))
seed = st.number_input(
    "Seed",
    min_value=0,
    max_value=999999,
    value=42,
    step=1,
)

if uploaded is not None:
    st.image(uploaded, caption="Εικόνα που ανέβασες", width=300)

if st.button("Δημιουργία και ανακατασκευή", disabled=uploaded is None):
    with st.status("Δημιουργία παζλ...", expanded=True) as status:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "uploaded_image.png"
            image_path.write_bytes(uploaded.getvalue())

            original, pieces = puzzle.generate_puzzle(
                 str(image_path), rows, cols, int(seed)
            )
            shuffled = puzzle.make_preview(pieces, rows, cols)
            st.write(f"Δημιουργήθηκαν {rows * cols} ανακατεμένα κομμάτια.")

            piece_arrays = [
                np.asarray(piece.image, dtype=np.float32) / 255.0
                for piece in pieces
            ]

            solver.FEATURE_MODE = mode
            st.write(f"Ανακατασκευή με μέθοδο: {mode}")
            placement, cost = solver.solve(piece_arrays, rows, cols)

            result_path = Path(temporary) / "reconstructed.png"
            solver.save_reconstruction(
                piece_arrays, placement, str(result_path), rows, cols
            )

            with Image.open(result_path) as image:
                reconstructed = image.copy()

        status.update(label="Η ανακατασκευή ολοκληρώθηκε", state="complete")

    columns = st.columns(3)
    columns[0].image(original, caption="Αρχική εικόνα", width="stretch")
    columns[1].image(shuffled, caption="Ανακατεμένα κομμάτια", width="stretch")
    columns[2].image(
        reconstructed,
        caption="Ανακατασκευή",
        width="stretch",
    )

    truth_grid = [0] * len(pieces)
    for index, piece in enumerate(pieces):
        position = piece.original_row * cols + piece.original_col
        truth_grid[position] = index

    solved_grid = [index for index, turns in placement]
    positions = sum(
        a == b for a, b in zip(solved_grid, truth_grid)
    )
    rotations = sum(
        turns == (-pieces[index].applied_rotation) % 4
        for index, turns in placement
    )
    true_neighbors = neighbor_pairs(truth_grid, rows, cols)
    recovered_neighbors = len(
        true_neighbors & neighbor_pairs(solved_grid, rows, cols)
    )

    count = rows * cols
    neighbor_count = rows * (cols - 1) + cols * (rows - 1)
    
    st.subheader("Απόλυτη αξιολόγηση")
    metrics = st.columns(3)
    metrics[0].metric("Σωστές θέσεις", f"{positions}/{count}")
    metrics[1].metric(
        "Σωστοί γείτονες", f"{recovered_neighbors}/{neighbor_count}"
    )
    metrics[2].metric("Σωστές περιστροφές", f"{rotations}/{count}")

    exact = np.array_equal(
        np.asarray(original),
        np.asarray(reconstructed),
    )
    st.write(f"Ακριβής ταύτιση εικόνας: **{'Ναι' if exact else 'Όχι'}**")
    st.caption(
        "Μια συνολική περιστροφή του παζλ διατηρεί τις γειτνιάσεις, "
        "αλλά μειώνει τις απόλυτες μετρήσεις θέσης και περιστροφής."
    )
    best = None
    grid_positions = np.arange(len(pieces)).reshape(rows, cols)

    for ccw_turns in ((0, 1, 2, 3) if rows == cols else (0, 2)):
        rotated_positions = np.rot90(
            grid_positions, k=ccw_turns
        ).ravel()
        aligned = [placement[int(i)] for i in rotated_positions]

        aligned_grid = [index for index, turns in aligned]
        aligned_positions = sum(
            a == b for a, b in zip(aligned_grid, truth_grid)
        )
        aligned_rotations = sum(
            (turns - ccw_turns) % 4
            == (-pieces[index].applied_rotation) % 4
            for index, turns in aligned
        )
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

    st.subheader("Αξιολόγηση έως συνολική περιστροφή")
    st.write(
        f"Στροφή ευθυγράμμισης: "
        f"**{turns * 90}° αριστερόστροφα**"
    )

    aligned_metrics = st.columns(3)
    aligned_metrics[0].metric(
        "Σωστές θέσεις", f"{aligned_positions}/{count}"
    )
    aligned_metrics[1].metric(
        "Σωστοί γείτονες", f"{aligned_neighbors}/{neighbor_count}"
    )
    aligned_metrics[2].metric(
        "Σωστές περιστροφές", f"{aligned_rotations}/{count}"
    )

    st.caption(
        "Η ευθυγράμμιση χρησιμοποιεί τη γνωστή σωστή λύση "
        "μόνο για μέτρηση. Δεν επηρεάζει τον αλγόριθμο ανακατασκευής."
    )

    buffer = BytesIO()
    reconstructed.save(buffer, format="PNG")
    st.download_button(
        "Λήψη ανακατασκευασμένης εικόνας",
        data=buffer.getvalue(),
        file_name="reconstructed.png",
        mime="image/png",
    )