import numpy as np
import pytest
from pixelproof.spatial_evaluation import evaluate_triplet, regions


def mask():
    out = np.zeros((20, 20), bool)
    out[5:15, 5:15] = True
    return out


def evaluate(a, c, edited, target=None):
    return evaluate_triplet(a, c, edited, mask() if target is None else target,
                            threshold=.5, boundary_width=2)


def test_partition_has_no_missing_or_overlapping_pixels():
    r = regions(mask(), 2)
    np.testing.assert_array_equal(sum(x.astype(int) for x in r.values()), np.ones((20, 20)))
    assert {k: int(v.sum()) for k, v in r.items()} == {
        'interior': 36, 'inner_boundary': 64, 'outer_boundary': 96, 'background': 204}


def test_image_edge_does_not_create_a_fictitious_boundary():
    target = mask(); target[:10, :10] = True
    r = regions(target, 2)
    assert r['interior'][0, 0]
    assert not r['inner_boundary'][0, 0]


def test_perfect_localizer_and_clean_negatives():
    z = np.zeros((20, 20))
    result = evaluate(z, z, mask().astype(float))
    assert result['maps']['ai_composite']['iou'] == 1
    assert result['maps']['ai_composite']['interior_background_ranking']['auc'] == 1
    for name in ('authentic', 'classical_edit'):
        assert result['maps'][name]['flagged_area_fraction'] == 0
        assert result['maps'][name]['all_pixel_ranking']['auc'] is None
        assert result['maps'][name]['iou'] is None  # Do not award perfect segmentation for empty truth.


def test_seam_only_detector_cannot_hide_missing_interior_or_classic_false_alarms():
    r = regions(mask(), 2)
    seam = (r['inner_boundary'] | r['outer_boundary']).astype(float)
    result = evaluate(np.zeros((20, 20)), seam, seam)
    ai = result['maps']['ai_composite']
    classic = result['maps']['classical_edit']
    assert ai['regions']['inner_boundary']['flagged_fraction'] == 1
    assert ai['regions']['interior']['flagged_fraction'] == 0
    assert ai['interior_background_ranking']['auc'] == .5
    assert classic['confusion_pixels']['tp'] == 0
    assert classic['confusion_pixels']['fp'] == 160


def test_thin_edit_retains_missing_interior_instead_of_fake_perfect_result():
    target = np.zeros((20, 20), bool); target[10, 3:17] = True
    z = np.zeros((20, 20)); result = evaluate(z, z, target.astype(float), target)
    assert not result['interior_and_background_available']
    assert result['maps']['ai_composite']['regions']['interior']['flagged_fraction'] is None
    assert result['maps']['ai_composite']['interior_background_ranking']['auc'] is None


@pytest.mark.parametrize('bad', [np.zeros((20, 19)), np.full((20, 20), np.nan),
                                np.ones((20, 20)) * 1.01, np.zeros((20, 20), int)])
def test_bad_maps_fail_closed(bad):
    z = np.zeros((20, 20))
    with pytest.raises(ValueError): evaluate(z, bad, z)


@pytest.mark.parametrize('bad', [0, -1, True, 1.5])
def test_invalid_boundary_width(bad):
    with pytest.raises(ValueError): regions(mask(), bad)


def test_mask_cannot_be_silently_thresholded_or_resized():
    with pytest.raises(ValueError): regions(mask().astype(np.uint8) * 255, 2)


def test_threshold_ties_are_positive_without_optimizing_on_truth():
    scores = np.full((20, 20), .5)
    result = evaluate(scores, scores, scores)
    assert result['maps']['authentic']['flagged_area_fraction'] == 1
    assert result['maps']['ai_composite']['iou'] == .25
    assert result['maps']['ai_composite']['all_pixel_ranking']['auc'] == .5


@pytest.mark.parametrize('threshold', [0, 1, float('nan'), True])
def test_invalid_threshold(threshold):
    z = np.zeros((20, 20))
    with pytest.raises(ValueError):
        evaluate_triplet(z, z, z, mask(), threshold=threshold, boundary_width=2)


def test_rejected_generation_keeps_negative_controls_without_fabricating_ai_metrics():
    z = np.zeros((20, 20))
    result = evaluate(z, z, None)
    assert not result['composite_available']
    assert set(result['maps']) == {'authentic', 'classical_edit'}
