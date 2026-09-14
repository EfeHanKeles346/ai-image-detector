import io
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pixelproof.e92_demo import AI_CUT, REAL_CUT, guarded_outcome, social_view
from pixelproof.research_serve import create_app


def photo(size=(256, 256), mode='RGB', color=128, **options):
    stream = io.BytesIO()
    Image.new(mode, size, color).save(stream, format='PNG', **options)
    return stream.getvalue()


class Engine:
    def analyze(self, picture):
        return 'ai_signal', 'stable_signal'


@pytest.mark.parametrize('scores,reference,expected', [
    ([AI_CUT, 1], [0, 0], 'ai_signal'),
    ([AI_CUT, AI_CUT - 1e-10], [0, 0], 'uncertain'),
    ([0, 0], [0, AI_CUT], 'uncertain'),
    ([0, REAL_CUT], [0, 0], 'uncertain'),
    ([0, REAL_CUT - 1e-10], [0, 0], 'no_clear_signal'),
    ([0, 0], [1, 1], 'uncertain'),
])
def test_guard_boundary_and_reference_cannot_accuse(scores, reference, expected):
    assert guarded_outcome(scores, reference)[0] == expected


@pytest.mark.parametrize('scores,reference', [([float('nan'), 1], [0, 0]), ([0], [0, 0]),
    ([0, 0], [0, float('inf')]), ([-1, 0], [0, 0]), ([0, 1.1], [0, 0])])
def test_guard_fails_closed(scores, reference):
    with pytest.raises(ValueError):
        guarded_outcome(scores, reference)


def test_guard_can_only_remove_positive_indications():
    rng = np.random.default_rng(93)
    for scores, reference in zip(rng.random((100, 2)), rng.random((100, 2))):
        result, _ = guarded_outcome(scores, reference)
        if result == 'ai_signal':
            assert all(scores >= AI_CUT)
        assert result != 'real'


def test_social_processing_matches_historical_file_transform(tmp_path):
    from experiments.e65_diagnostic import social_q75_bytes
    im = Image.fromarray(np.random.default_rng(1).integers(0, 256, (1200, 1700, 3), dtype=np.uint8))
    path = tmp_path / 'test.png'
    im.save(path)
    with Image.open(io.BytesIO(social_q75_bytes(path))) as expected:
        np.testing.assert_array_equal(np.asarray(social_view(im)), np.asarray(expected.convert('RGB')))


def test_api_has_e92_identity_and_no_legacy_or_probability_payload():
    with TestClient(create_app(Engine)) as client:
        result = client.post('/analyze', content=photo(), headers={'content-type': 'image/png'})
        assert result.status_code == 200
        payload = result.json()
        assert payload['model_id'] == 'E92'
        assert payload['outcome'] == 'ai_signal'
        assert payload['research_only'] is True
        assert 'p_ai' not in payload and 'tile_map' not in payload
        assert client.get('/health').json()['status'] == 'ready'


def test_invalid_and_tiny_inputs_never_run_inference():
    class Never:
        def analyze(self, picture):
            pytest.fail('Input guard must run before inference')
    with TestClient(create_app(Never)) as client:
        for raw, expected in [(b'garbage', 415), (b'', 422), (photo((223, 224)), 200),
                              (photo(mode='RGBA', color=(0, 0, 0, 0)), 415),
                              (photo((5000, 4000)), 413)]:
            r = client.post('/analyze', content=raw, headers={'content-type': 'image/png'})
            assert r.status_code == expected
            if expected == 200:
                assert r.json()['outcome'] == 'uncertain'
                assert r.json()['reason'] == 'image_too_small'
        stream = io.BytesIO()
        Image.new('RGB', (224, 224)).save(stream, format='PNG', save_all=True,
            append_images=[Image.new('RGB', (224, 224), 'white')], duration=100, loop=0)
        assert client.post('/analyze', content=stream.getvalue(), headers={'content-type': 'image/png'}).status_code == 415


def test_chunked_oversize_upload_is_rejected_before_decode():
    def chunks():
        yield b'x' * (6 * 1024**2)
        yield b'x' * (7 * 1024**2)
    with TestClient(create_app(Engine)) as client:
        assert client.post('/analyze', content=chunks(), headers={'content-type': 'image/png'}).status_code == 413
        assert client.post('/analyze', content=photo(), headers={'content-type': 'image/png'}).status_code == 200


def test_load_failure_never_falls_back_or_exposes_paths():
    def missing():
        raise FileNotFoundError('/private/secret/checkpoint')
    with TestClient(create_app(missing)) as client:
        assert client.get('/health').json()['status'] == 'unavailable'
        r = client.post('/analyze', content=photo(), headers={'content-type': 'image/png'})
        assert r.status_code == 503
        assert '/private' not in r.text


def test_runtime_failure_does_not_leave_stale_result_or_hold_slot():
    class Fails:
        def analyze(self, picture):
            raise ValueError('nonfinite model score /private/path')
    with TestClient(create_app(Fails)) as client:
        for _ in range(2):
            r = client.post('/analyze', content=photo(), headers={'content-type': 'image/png'})
            assert r.status_code == 503
            assert 'outcome' not in r.json() and '/private' not in r.text


def test_timeout_holds_slot_until_actual_worker_completion():
    done = threading.Event()
    class Slow:
        def analyze(self, picture):
            assert done.wait(3)
            return 'uncertain', 'inconsistent_or_borderline'
    with TestClient(create_app(Slow, inference_timeout=0.01)) as client:
        try:
            r = client.post('/analyze', content=photo(), headers={'content-type': 'image/png'})
            assert r.status_code == 504
            busy = client.post('/analyze', content=photo(), headers={'content-type': 'image/png'})
            assert busy.status_code == 429
        finally:
            done.set()
        for _ in range(100):
            time.sleep(.005)
            r = client.post('/analyze', content=photo(), headers={'content-type': 'image/png'})
            if r.status_code != 429:
                break
        assert r.status_code == 200


def test_cors_is_explicit():
    with TestClient(create_app(Engine)) as client:
        for origin, allowed in [('http://localhost:3000', True), ('https://unrelated.example', False)]:
            r = client.options('/analyze', headers={'origin': origin,
                'access-control-request-method': 'POST', 'access-control-request-headers': 'content-type'})
            assert ('access-control-allow-origin' in r.headers) is allowed
