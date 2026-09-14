"""Targeted local HTTP integration replay of existing consumed results, not an eval."""
import json
import time
import urllib.request
from pathlib import Path
from PIL import Image

from pixelproof.e92_demo import CANDIDATE_SHA, digest
from pixelproof.project_paths import ML_ROOT, WORK_ROOT


def main():
    rows = json.loads((WORK_ROOT / 'e94_display_rows.json').read_text())
    eligible = [r for r in rows if r['input_eligibility'] == 'accepted']
    # Cover each existing result/warning combination; selection is deliberately
    # outcome-based for integration testing and cannot estimate performance.
    groups = sorted({(r['label'], r['outcome'], r['review_required']) for r in eligible})
    selected = [min((r for r in eligible if (r['label'], r['outcome'], r['review_required']) == key),
                    key=lambda r: (r['sha256'], r['parent_id'])) for key in groups]
    with urllib.request.urlopen('http://127.0.0.1:8800/health', timeout=30) as response:
        health = json.load(response)
    assert health['status'] == 'ready' and health['model_id'] == 'E92'
    assert health['display_policy'] == 'e92-preserve-alerts-v1'
    assert health['artifact_sha256'] == CANDIDATE_SHA
    results = []
    for row in selected:
        path = Path(row['path'])
        assert digest(path) == row['sha256']
        with Image.open(path) as im:
            mime = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp'}[im.format]
        request = urllib.request.Request('http://127.0.0.1:8800/analyze', data=path.read_bytes(),
            headers={'Content-Type': mime, 'Origin': 'http://localhost:3002'}, method='POST')
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=100) as response:
            assert response.headers.get('Access-Control-Allow-Origin') == 'http://localhost:3002'
            payload = json.load(response)
        elapsed = time.monotonic() - started
        for key in ['outcome', 'reason', 'guard_outcome', 'review_required', 'display_policy']:
            assert payload[key] == row[key], (key, row['parent_id'])
        assert payload['model_id'] == 'E92' and payload['artifact_sha256'] == CANDIDATE_SHA
        results.append({'parent_id': row['parent_id'], 'sha256': row['sha256'], 'label': row['label'],
                        'response': payload, 'http_seconds': elapsed})
    receipt = {'state': 'local_HTTP_integration_passed', 'health': health, 'cases': results,
               'api_code_sha256': digest(ML_ROOT / 'src/pixelproof/internship_serve.py'),
               'limits': 'Outcome-selected engineering replay on consumed originals. Includes local HTTP, decode, paired native inference. No broad latency/accuracy claim; no new model selection.'}
    with (ML_ROOT.parent / 'evidence/e94_http_smoke.json').open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps({'state': receipt['state'], 'cases': len(results),
                      'seconds': [r['http_seconds'] for r in results]}, indent=2))


if __name__ == '__main__':
    main()
