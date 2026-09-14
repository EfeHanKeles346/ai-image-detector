import math

import pytest
from pydantic import ValidationError

from pixelproof.demo_scores import scored_display_result
from pixelproof.internship_serve import DemoResult
from pixelproof.e92_demo import AI_CUT, REAL_CUT


@pytest.mark.parametrize('scores,reference,outcome', [
    ([AI_CUT, 0], [0, 0], 'ai_signal'),
    ([math.nextafter(AI_CUT, 0), AI_CUT], [0, 0], 'uncertain'),
    ([0, 0], [AI_CUT, 0], 'uncertain'),
    ([REAL_CUT, 0], [0, 0], 'uncertain'),
    ([0, 0], [0, 0], 'no_clear_signal'),
    ([1, 1], [1, 1], 'ai_signal'),
])
def test_presentation_preserves_boundaries_without_averaging(scores, reference, outcome):
    result = DemoResult(**scored_display_result(scores, reference), width=256, height=256)
    assert result.schema_version == 3
    assert result.outcome == outcome
    assert result.model_score.original == scores[0]
    assert result.model_score.social_q75 == scores[1]
    assert result.model_score.calibrated is False


@pytest.mark.parametrize('score', [math.nan, math.inf, -1, 1.01])
def test_invalid_raw_scores_cannot_be_displayed(score):
    with pytest.raises(ValueError):
        scored_display_result([score, .1], [.1, .1])


def test_response_rejects_missing_scores_or_fabricated_probability():
    payload = scored_display_result([.9, .9], [.1, .1])
    for score in [None, {**payload['model_score'], 'calibrated': True},
                  {**payload['model_score'], 'original': 0},
                  {**payload['model_score'], 'social_q75': 0},
                  {**payload['model_score'], 'original': math.nan}]:
        with pytest.raises(ValidationError):
            DemoResult(**{**payload, 'model_score': score}, width=256, height=256)
    small = DemoResult(outcome='uncertain', reason='image_too_small', guard_outcome='not_run',
                       review_required=True, width=223, height=224)
    assert small.model_score is None
    with pytest.raises(ValidationError):
        DemoResult(**{**small.model_dump(), 'model_score': payload['model_score']})
