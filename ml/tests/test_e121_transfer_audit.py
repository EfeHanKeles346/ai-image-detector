import pytest
from experiments.e121_context_transfer_audit import center_fraction,summarize


def test_center_window_area_is_not_full_frame_or_crop_union():
    assert center_fraction(1000,1000)==pytest.approx(.765625)
    assert center_fraction(2000,1000)==pytest.approx(.3828125)
    assert center_fraction(1000,2000)==center_fraction(2000,1000)
    with pytest.raises(ValueError):center_fraction(0,1000)


def test_transfer_summary_detects_swapped_errors_and_unchanged_counts():
    rows=[{'condition':'original','label':1,'source':'AI','E103_score':old,'score':new}
        for old,new in ((.1,.01),(.01,.1))]
    result=summarize(rows)[0]
    assert result['changed_AI_decisions']==2
    assert result['nonnegative_correction_views']==1
