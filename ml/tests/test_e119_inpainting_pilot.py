import numpy as np
from PIL import Image
import pytest
from experiments.e119_inpainting_pilot import composite, make_mask, select, SENSORS


def test_exact_background_control_and_deterministic_masks():
    original = Image.new('RGB', (512,512), (20,30,40)); generated = Image.new('RGB',(512,512),(100,110,120))
    for shape in ('rectangle','ellipse'):
        mask = make_mask('train:example', shape)
        assert np.array_equal(mask, make_mask('train:example', shape))
        m = np.asarray(mask)>0; assert .05 < m.mean() < .3
        result = np.asarray(composite(original, generated, mask))
        assert np.array_equal(result[~m], np.asarray(original)[~m])
        assert np.array_equal(result[m], np.asarray(generated)[m])
    with pytest.raises(ValueError): composite(original, generated, Image.new('L',(512,512),255))
    with pytest.raises(ValueError): composite(original, generated.resize((256,256)), mask)


def test_pilot_parent_selection_has_no_evaluation_role_fallback():
    rows = [{'source': 'MIDD:'+s, 'parent_id': s+str(i), 'sha256': s+str(i), 'scene_group': s+str(i),
        'role': 'TRAIN', 'label': 0, 'training_allowed': True, 'width': 512, 'height': 512} for s in SENSORS for i in range(2)]
    assert len(select(rows)) == 16
    assert select(rows) == select(rows[::-1])
    rows[0]['role'] = 'DEVELOPMENT'
    with pytest.raises(ValueError, match='TRAIN'): select(rows)
