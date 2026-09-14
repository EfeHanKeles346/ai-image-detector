import numpy as np
from PIL import Image
import pytest
from experiments.e71_features import aggregate
from experiments.e104_context_audit import context_coordinates, selected_indices
from pixelproof.e32_candidate import standardized_array


def test_context_role_is_not_recoverable_from_symmetric_pooling():
    raw = np.zeros((3, 3, 768), dtype=np.float32)
    raw[:, 0] = 1; raw[:, 1] = 2; raw[:, 2] = 3
    swapped = raw[:, [1, 0, 2]]
    np.testing.assert_array_equal(aggregate(raw), aggregate(swapped))
    assert not np.array_equal(context_coordinates(raw), context_coordinates(swapped))
    # The two local texture slots have no fixed semantic order.
    np.testing.assert_array_equal(context_coordinates(raw), context_coordinates(raw[:, [0, 2, 1]]))


def test_historical_global_crop_cannot_observe_distant_frame_edges():
    original = np.zeros((200, 600, 3), dtype=np.uint8)
    changed = original.copy(); changed[:, :90] = 255; changed[:, -90:] = 255
    np.testing.assert_array_equal(standardized_array(Image.fromarray(original)),
                                  standardized_array(Image.fromarray(changed)))


def test_selection_is_score_blind_and_train_only():
    rows = [{'parent_id': str(i), 'source': 'a' if i < 4 else 'b', 'label': int(i >= 4),
             'role': 'TRAIN', 'score': .99} for i in range(8)]
    chosen = {rows[i]['parent_id'] for i in selected_indices(rows)}
    reordered = [dict(r, score=0.) for r in reversed(rows)]
    assert chosen == {reordered[i]['parent_id'] for i in selected_indices(reordered)}
    rows[0]['role'] = 'DEV'
    with pytest.raises(ValueError, match='TRAIN'):
        selected_indices(rows)


def test_invalid_context_vectors_fail():
    for value in (np.zeros((1, 2, 3), dtype=np.float32), np.zeros((1, 3, 3)),
                  np.full((1, 3, 3), np.nan, dtype=np.float32)):
        with pytest.raises(ValueError, match='float32'):
            context_coordinates(value)
