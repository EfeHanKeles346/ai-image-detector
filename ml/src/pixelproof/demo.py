"""Preflight, smoke-test and run the local PixelProof model demo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


ML_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ML_ROOT.parent
DEFAULT_IMAGE = ML_ROOT / "artifacts/figures/generators.png"
EXPECTED_ARTIFACT_ID = "e20-tile-resnet18-seed2024"


class DemoError(RuntimeError):
    """A local demo prerequisite or runtime contract is not satisfied."""


def _run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise DemoError(f"required command is missing: {command[0]}") from None


def _require_success(label: str, result: subprocess.CompletedProcess[str], fix: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise DemoError(f"{label} failed: {detail or 'unknown error'}. {fix}")
    return result.stdout.strip()


def _check_port(port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
            stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            stream.bind(("127.0.0.1", port))
    except OSError as error:
        raise DemoError(
            f"port {port} is unavailable on 127.0.0.1 ({error}); stop the process using it "
            "or choose another --api-port/--web-port"
        ) from error


def check_environment(api_port: int = 8799, web_port: int = 3000) -> list[str]:
    """Fail with a repair instruction at the first unmet local-demo prerequisite."""
    if api_port == web_port:
        raise DemoError("API and web ports must be different")
    if not 1 <= api_port <= 65535 or not 1 <= web_port <= 65535:
        raise DemoError("ports must be integers from 1 through 65535")

    results = []
    python_version = _require_success(
        "virtualenv Python",
        _run([sys.executable, "--version"]),
        "Recreate ml/.venv with Python 3.13.",
    )
    results.append(f"python: {python_version}")

    imports = _require_success(
        "Python dependency import",
        _run([
            sys.executable,
            "-c",
            "import fastapi, numpy, PIL, sklearn, torch, torchvision, uvicorn",
        ], cwd=ML_ROOT),
        "Run ml/.venv/bin/pip install -r ml/requirements-serving.lock.",
    )
    if imports:
        results.append(f"python imports: {imports}")
    results.append("python imports: ready")

    _require_success(
        "Python dependency check",
        _run([sys.executable, "-m", "pip", "check"], cwd=ML_ROOT),
        "Reinstall ml/requirements-serving.lock in ml/.venv.",
    )
    results.append("python dependency graph: ready")

    artifact_output = _require_success(
        "canonical model verification",
        _run([
            sys.executable,
            "-m",
            "pixelproof.artifact_registry",
            "check",
            "--group",
            "project_model",
        ], cwd=ML_ROOT),
        "Place the canonical checkpoint as documented in ml/ARTIFACTS.md.",
    )
    artifact_report = json.loads(artifact_output)
    if artifact_report.get("checked") != [EXPECTED_ARTIFACT_ID]:
        raise DemoError(
            f"artifact registry did not verify {EXPECTED_ARTIFACT_ID}: {artifact_report}"
        )
    results.append(f"project artifact: {EXPECTED_ARTIFACT_ID} verified")

    missing_scripts = [
        name for name in ("pixelproof-predict", "pixelproof-evaluate-project")
        if not (ML_ROOT / ".venv/bin" / name).is_file()
    ]
    if missing_scripts:
        raise DemoError(
            f"installed project commands are stale ({', '.join(missing_scripts)} missing); "
            "run ml/.venv/bin/pip install --no-deps -e ml"
        )
    results.append("project CLI entry points: ready")

    node_path = shutil.which("node")
    npm_path = shutil.which("npm")
    if not node_path or not npm_path:
        raise DemoError("Node.js and npm are required; install Node.js 22.13 or newer")
    node_version = _require_success(
        "Node.js",
        _run([node_path, "--version"]),
        "Install Node.js 22.13 or newer.",
    )
    try:
        node_parts = tuple(int(part) for part in node_version.lstrip("v").split(".")[:3])
    except (TypeError, ValueError):
        raise DemoError(f"could not parse Node.js version: {node_version!r}") from None
    if len(node_parts) != 3 or node_parts < (22, 13, 0):
        raise DemoError(f"Node.js {node_version} is unsupported; install 22.13 or newer")
    results.append(f"node: {node_version}")

    if not (REPO_ROOT / "node_modules").is_dir():
        raise DemoError("web dependencies are missing; run npm ci in the repository root")
    _require_success(
        "npm dependency check",
        _run([npm_path, "ls", "--depth=0"], cwd=REPO_ROOT),
        "Run npm ci in the repository root.",
    )
    results.append("web dependency graph: ready")

    _check_port(api_port)
    _check_port(web_port)
    results.append(f"ports: {api_port} and {web_port} available on loopback")
    return results


def _request_json(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise DemoError(f"HTTP {error.code} from {request.full_url}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise DemoError(f"cannot reach {request.full_url}: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DemoError(f"invalid JSON from {request.full_url}: {error}") from error
    if not isinstance(payload, dict):
        raise DemoError(f"expected a JSON object from {request.full_url}")
    return payload


def validate_health(payload: dict[str, Any]) -> None:
    if payload.get("status") != "ready" or payload.get("project_model_ready") is not True:
        raise DemoError(f"project model is not ready: {payload.get('load_errors', payload)}")
    model = payload.get("project_model")
    if not isinstance(model, dict) or model.get("artifact_id") != EXPECTED_ARTIFACT_ID:
        raise DemoError(f"health returned the wrong project model: {model}")
    digest = model.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise DemoError("health did not return the verified 64-character artifact SHA-256")


def validate_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    project = payload.get("project_model")
    if payload.get("method") != "project_model" or not isinstance(project, dict):
        raise DemoError("prediction did not use the canonical project_model path")
    if payload.get("verdict") not in {"ai", "uncertain"}:
        raise DemoError("project model returned an invalid or authenticity-certifying verdict")
    if project.get("research_only") is not True:
        raise DemoError("prediction omitted the research_only contract")
    for field in ("score", "threshold"):
        value = project.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise DemoError(f"prediction returned an invalid {field}: {value!r}")
    digest = project.get("artifact_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise DemoError("prediction omitted the verified artifact SHA-256")
    if not isinstance(project.get("tile_count"), int) or project["tile_count"] <= 0:
        raise DemoError("prediction returned an invalid tile count")
    return project


def _multipart(image: Path) -> tuple[bytes, str]:
    try:
        raw = image.read_bytes()
    except OSError as error:
        raise DemoError(f"cannot read smoke image {image}: {error}") from error
    if not raw:
        raise DemoError(f"smoke image is empty: {image}")
    boundary = f"pixelproof-{uuid.uuid4().hex}"
    filename = image.name.replace('"', "")
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="method"\r\n\r\nproject_model\r\n',
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: image/png\r\n\r\n",
        raw,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return body, boundary


def smoke_api(api_url: str, image: Path = DEFAULT_IMAGE, timeout: float = 30.0) -> dict[str, Any]:
    origin = api_url.rstrip("/")
    health = _request_json(urllib.request.Request(f"{origin}/health"), timeout)
    validate_health(health)
    body, boundary = _multipart(image)
    prediction = _request_json(urllib.request.Request(
        f"{origin}/predict",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    ), timeout)
    project = validate_prediction(prediction)
    return {
        "status": "ok",
        "image": str(image.resolve()),
        "score": project["score"],
        "threshold": project["threshold"],
        "triggered": project["triggered"],
        "tile_count": project["tile_count"],
        "artifact_sha256": project["artifact_sha256"],
    }


def _wait_for_api(url: str, process: subprocess.Popen[Any], timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not started"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise DemoError(f"API process exited early with code {process.returncode}")
        try:
            payload = _request_json(urllib.request.Request(f"{url}/health"), 2.0)
            validate_health(payload)
            return
        except DemoError as error:
            last_error = str(error)
        time.sleep(0.5)
    raise DemoError(f"API did not become ready within {timeout:g}s: {last_error}")


def _wait_for_web(url: str, process: subprocess.Popen[Any], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not started"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise DemoError(f"web process exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                html = response.read().decode("utf-8")
            if response.status == 200 and "PixelProof" in html and "E20 ResNet-18" in html:
                return
            last_error = f"HTTP {response.status} without the PixelProof E20 shell"
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise DemoError(f"web UI did not become ready within {timeout:g}s: {last_error}")


def _stop(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait(timeout=5)


def start_demo(
    api_port: int,
    web_port: int,
    image: Path,
    r1b_data_root: Path | None = None,
) -> None:
    for result in check_environment(api_port, web_port):
        print(f"check: {result}")

    api_url = f"http://127.0.0.1:{api_port}"
    web_url = f"http://127.0.0.1:{web_port}"
    api_env = os.environ.copy()
    api_env["PYTHONPATH"] = str(ML_ROOT / "src")
    api_env["PIXELPROOF_CORS_ORIGINS"] = web_url
    api_env["PIXELPROOF_RUNTIME_PROFILE"] = "demo"
    if r1b_data_root is not None:
        root = r1b_data_root.expanduser().resolve()
        expected = root / "e32/models/e32_r1b_cf.joblib"
        if not expected.is_file():
            raise DemoError(f"R1b artifact is missing below the selected data root: {expected}")
        api_env["PIXELPROOF_DATA_ROOT"] = str(root)
        api_env["PIXELPROOF_R1B"] = "1"
    web_env = os.environ.copy()
    web_env["NEXT_PUBLIC_PIXELPROOF_API_URL"] = api_url
    api = None
    web = None
    try:
        print(f"starting API: {api_url}")
        api = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "pixelproof.serve:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
            ],
            cwd=ML_ROOT,
            env=api_env,
            start_new_session=True,
        )
        _wait_for_api(api_url, api)
        if r1b_data_root is not None:
            health = _request_json(urllib.request.Request(f"{api_url}/health"), 10.0)
            if health.get("r1b_research_ready") is not True:
                raise DemoError(
                    f"R1b research signal did not load: {health.get('load_errors', health)}"
                )
        smoke = smoke_api(api_url, image)
        print(
            f"smoke: score={smoke['score']:.4f}, threshold={smoke['threshold']:.4f}, "
            f"tiles={smoke['tile_count']}, hash={smoke['artifact_sha256'][:12]}..."
        )

        print(f"starting web UI: {web_url}")
        web = subprocess.Popen(
            [
                "npm",
                "run",
                "dev",
                "--",
                "--hostname",
                "127.0.0.1",
                "--port",
                str(web_port),
            ],
            cwd=REPO_ROOT,
            env=web_env,
            start_new_session=True,
        )
        _wait_for_web(web_url, web)
        print(f"PixelProof is ready: {web_url}")
        print("Press Ctrl+C to stop both local processes.")
        while True:
            if api.poll() is not None:
                raise DemoError(f"API process exited with code {api.returncode}")
            if web.poll() is not None:
                raise DemoError(f"web process exited with code {web.returncode}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopping PixelProof...")
    finally:
        _stop(web)
        _stop(api)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="verify local demo prerequisites")
    check_parser.add_argument("--api-port", type=int, default=8799)
    check_parser.add_argument("--web-port", type=int, default=3000)

    smoke_parser = subparsers.add_parser("smoke", help="validate a running API with one image")
    smoke_parser.add_argument("--api-url", default="http://127.0.0.1:8799")
    smoke_parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)

    start_parser = subparsers.add_parser("start", help="check, smoke and run API plus web UI")
    start_parser.add_argument("--api-port", type=int, default=8799)
    start_parser.add_argument("--web-port", type=int, default=3000)
    start_parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    start_parser.add_argument(
        "--r1b-data-root",
        type=Path,
        help="enable the frozen R1b research card from this PixelProof dataset root",
    )
    args = parser.parse_args()

    try:
        if args.command == "check":
            for result in check_environment(args.api_port, args.web_port):
                print(f"check: {result}")
            print("PixelProof local demo prerequisites are ready.")
        elif args.command == "smoke":
            print(json.dumps(smoke_api(args.api_url, args.image), indent=2))
        else:
            start_demo(args.api_port, args.web_port, args.image, args.r1b_data_root)
    except DemoError as error:
        parser.exit(2, f"PixelProof demo error: {error}\n")


if __name__ == "__main__":
    main()
