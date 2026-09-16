import math
import pytest
from pydantic import ValidationError
from pixelproof.demo_scores import scored_display_result
from pixelproof.e92_demo import AI_CUT, REAL_CUT
from pixelproof.primary_demo_policy import primary_result
from pixelproof.internship_serve import DemoResult, create_app
from fastapi.testclient import TestClient
from test_internship_serve import photo


def test_old_veto_is_advisory_but_historical_policy_remains_reproducible():
    s = [.0001, .0039]
    assert scored_display_result(s, [AI_CUT, 0])['outcome'] == 'uncertain'
    new = primary_result(s, [AI_CUT, 0])
    assert new['outcome'] == 'no_clear_signal'
    assert new['reference_ai_warning'] is True and new['review_required'] is False
    assert new['model_score'] == scored_display_result(s, [AI_CUT, 0])['model_score']


def test_reference_warning_does_not_change_any_primary_decision():
    values = [0., math.nextafter(REAL_CUT, 0), REAL_CUT, math.nextafter(AI_CUT, 0), AI_CUT, 1.]
    for first in values:
        for second in values:
            baseline = primary_result([first, second], [0, 0])
            warned = primary_result([first, second], [1, 1])
            for key in ['outcome', 'reason', 'review_required', 'guard_outcome', 'model_score']:
                assert baseline[key] == warned[key]
            assert (warned['outcome'] == 'ai_signal') == (first >= AI_CUT)
            DemoResult(**warned, width=256, height=256)


def test_api_rejects_stale_veto_result_and_reports_advisory_separately():
    result = primary_result([.0001, .0039], [.9, .1])
    with pytest.raises(ValidationError):
        DemoResult(**{**result, 'outcome': 'uncertain'}, width=256, height=256)
    with pytest.raises(ValidationError):
        DemoResult(**{**result, 'reference_ai_warning': None}, width=256, height=256)
    class Engine:
        def analyze(self, image): return result
    with TestClient(create_app(Engine)) as client:
        payload = client.post('/analyze', content=photo(), headers={'content-type':'image/png'}).json()
        assert payload['schema_version'] == 4
        assert payload['outcome'] == 'no_clear_signal' and payload['reference_ai_warning']
        assert client.get('/health').json()['display_policy'] == result['display_policy']
