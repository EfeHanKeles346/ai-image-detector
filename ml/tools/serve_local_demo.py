"""Start the existing E92 API offline, with explicit disk and port-3002 CORS.

Use the existing virtualenv. No installer, acquisition or model selection runs here.
Keep this command running; Ctrl-C stops only the child it started. An already-ready
exact E92 listener is reused, while any other listener is left untouched.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[2]
EXPECTED_SHA = '3a68c50d7cabd17d74c90bdcaf3b74aaacbc6c07e0bf28e332b1b91f99c9ef35'
ORIGINS = 'http://localhost:3002,http://127.0.0.1:3002'


def environment(data_root):
    root = Path(data_root).expanduser().resolve()
    manifest = json.loads((REPO / 'evidence/e93_runtime_manifest.json').read_text())
    missing = [str(root / name) for name in manifest['data_files'] if not (root / name).is_file()]
    if not root.is_dir() or missing:
        raise ValueError('E92 veri diski veya gerekli yerel dosyalar eksik; indirme yapılmadı.')
    env = os.environ.copy()
    env.update(PIXELPROOF_DATA_ROOT=str(root), PYTHONPATH=str(REPO / 'ml') + os.pathsep + str(REPO / 'ml/src'),
               PIXELPROOF_CORS_ORIGINS=ORIGINS, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return env


def health():
    # Never send local checks through a configured remote proxy or follow redirects.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs): return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        request = urllib.request.Request('http://127.0.0.1:8800/health', headers={'Origin':'http://localhost:3002'})
        with opener.open(request, timeout=2) as response:
            data = json.loads(response.read(8192))
            if not isinstance(data, dict): return None
            data['_cors_ok'] = response.headers.get('Access-Control-Allow-Origin') == 'http://localhost:3002'
            return data
    except (OSError, ValueError, urllib.error.URLError):
        return None


def ready(data):
    return isinstance(data, dict) and data.get('status') == 'ready' and data.get('model_id') == 'E92' and \
        data.get('schema_version') == 4 and \
        data.get('artifact_sha256') == EXPECTED_SHA and data.get('guard_id') == 'e92-paired-v2' and \
        data.get('display_policy') == 'e92-primary-reference-advisory-v2' and data.get('research_only') is True and \
        data.get('downloads_allowed') is False and data.get('_cors_ok') is True


def main(data_root, check_only=False):
    env = environment(data_root)
    with socket.socket() as probe:
        probe.settimeout(1); occupied = probe.connect_ex(('127.0.0.1', 8800)) == 0
    current = health() if occupied else None
    if occupied and not ready(current):
        raise RuntimeError('8800 portu kullanımda; beklenen E92/CORS doğrulanamadı. Mevcut süreç durdurulmadı.')
    if check_only or occupied:
        print(json.dumps({'local_files_present':True, 'exact_E92_ready':ready(current),
                          'reused_existing':occupied, 'downloads':0}))
        return 0 if (not check_only or ready(current)) else 1
    child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'pixelproof.internship_serve:app',
        '--host', '127.0.0.1', '--port', '8800'], cwd=REPO, env=env)
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if child.poll() is not None: raise RuntimeError('E92 açılışı başarısız; üstteki yerel kayıtları inceleyin.')
            if ready(health()):
                print('E92 hazır. Yerel demo: http://localhost:3002/ — durdurmak için Ctrl-C.', flush=True)
                return child.wait()
            time.sleep(.5)
        raise RuntimeError('E92 doğrulanmış hazır durumuna ulaşamadı; başlatılan süreç durduruluyor.')
    finally:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try: raise SystemExit(main(args.data_root, args.check_only))
    except KeyboardInterrupt: raise SystemExit(130)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr); raise SystemExit(1)
