import numpy as np
import pytest
from experiments.e127_inpaint_pixel_audit import image_metrics


def inputs():
    original = np.full((8, 8, 3), 10, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8); mask[2:6, 2:6] = 255
    return original, mask


def test_exact_composite_has_no_background_drift_but_measurable_boundary():
    original, mask = inputs(); edited = original.copy(); edited[mask > 0] = 30
    result = image_metrics(original, edited, mask)
    assert result['outside_mae'] == result['outside_changed_pixel_fraction'] == 0
    assert result['inside_mae'] == pytest.approx(20/255)
    assert result['boundary_edges'] == 16
    assert result['boundary_gradient'] == pytest.approx(20/255)
    assert result['boundary_gradient_original'] == 0
    assert result['inside_changed_pixel_fraction'] == 1


def test_every_changed_pixel_does_not_imply_large_background_drift():
    original, mask = inputs()
    result = image_metrics(original, original+1, mask)
    assert result['outside_changed_pixel_fraction'] == 1
    assert result['outside_mae'] == pytest.approx(1/255)
    assert result['outside_rmse'] == pytest.approx(1/255)
    assert result['boundary_gradient_delta'] == 0


@pytest.mark.parametrize('value', [1, 128, 255, 0])
def test_nonbinary_or_nonpartial_masks_cannot_produce_quality_report(value):
    original, mask = inputs(); mask[:] = value
    with pytest.raises(ValueError): image_metrics(original, original, mask)


def test_same_geometry_and_dtype_are_required():
    original, mask = inputs()
    with pytest.raises(ValueError): image_metrics(original, original[:, :4], mask)
    with pytest.raises(ValueError): image_metrics(original, original.astype(np.float32), mask)
