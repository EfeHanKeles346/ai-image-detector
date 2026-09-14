import copy
import numpy as np
from PIL import Image
import pytest
from experiments.e128_inpaint_context_inputs import context_image, texture_proxy, validate_population


def test_context_geometry_preserves_aspect_and_known_center_square():
    image = Image.fromarray(np.random.default_rng(128).integers(0, 256, (800, 1200, 3), dtype=np.uint8))
    result, geometry = context_image(image)
    expected = image.crop((200, 0, 1000, 800)).resize((512, 512), Image.Resampling.LANCZOS)
    assert np.array_equal(result, expected)
    assert geometry['native_center_square'] == [200, 0, 1000, 800]
    assert geometry['source_area_fraction'] == pytest.approx(2/3)
    assert geometry['linear_scale'] == .64


def test_native512_square_is_identity_and_small_originals_are_rejected():
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (512, 512, 3), dtype=np.uint8))
    result, geometry = context_image(image)
    assert np.array_equal(result, image) and not geometry['resized']
    with pytest.raises(ValueError): context_image(Image.new('RGB', (511, 700)))
    assert texture_proxy(Image.new('RGB', (512, 512), 'white')) == 0


def population():
    rows = [dict(parent_id=str(i), sha256=str(i), role='TRAIN', label=0, training_allowed=True) for i in range(16)]
    prepared = [dict(index=i, parent_id=str(i), original_sha256=str(i), role='TRAIN_RESEARCH_PILOT', seed=119000+i) for i in range(16)]
    return rows, prepared


@pytest.mark.parametrize('field,value', [('parent_id', 'different'), ('original_sha256', 'changed'),
    ('role', 'DEVELOPMENT'), ('seed', 0)])
def test_resizing_cannot_change_ancestry_role_or_seed(field, value):
    rows, prepared = population(); validate_population(rows, prepared)
    prepared[0][field] = value
    with pytest.raises(ValueError): validate_population(rows, prepared)


def test_failed_generation_cannot_be_replaced_or_omitted():
    rows, prepared = population()
    with pytest.raises(ValueError): validate_population(rows, prepared[:-1])
    reversed_order = copy.deepcopy(prepared); reversed_order.reverse()
    with pytest.raises(ValueError): validate_population(rows, reversed_order)
