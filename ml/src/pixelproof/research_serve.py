"""Local-only E92 student demo: raw image POST /analyze, no upload persistence.

Launch: PYTHONPATH=ml:ml/src PIXELPROOF_DATA_ROOT=/path/to/existing/data \
  ml/.venv/bin/python -m uvicorn pixelproof.research_serve:app --host 127.0.0.1 --port 8800
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import io
import logging
import os
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from pixelproof.e92_demo import CANDIDATE_SHA, E92Engine, GUARD_ID, MODEL_ID
from pixelproof.image_input import DEFAULT_LIMITS, ImagePolicyError, decode_image

logger = logging.getLogger(__name__)


class DemoResult(BaseModel):
    schema_version: Literal[2] = 2
    model_id: Literal['E92'] = MODEL_ID
    guard_id: Literal['e92-stability-v1'] = GUARD_ID
    artifact_sha256: Literal[CANDIDATE_SHA] = CANDIDATE_SHA
    research_only: Literal[True] = True
    outcome: Literal['ai_signal', 'no_clear_signal', 'uncertain']
    reason: Literal['stable_signal', 'limited_negative_evidence', 'inconsistent_or_borderline', 'image_too_small']
    width: int
    height: int


def decode_demo(raw: bytes):
    image = decode_image(raw)
    # Do not silently classify only the first frame of animated PNG/WebP, or
    # convert high-bit-depth/scientific images into ordinary camera-photo evidence.
    with Image.open(io.BytesIO(raw)) as original:
        if original.format != 'MPO' and getattr(original, 'is_animated', False):
            raise ImagePolicyError(415, 'Hareketli görseller desteklenmiyor. Tek bir fotoğraf seçin.')
        if original.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
            raise ImagePolicyError(415, 'Bu renk biçimi desteklenmiyor. Standart bir JPG veya PNG seçin.')
        if original.mode in ('RGBA', 'LA') or 'transparency' in original.info:
            if original.convert('RGBA').getchannel('A').getextrema()[0] < 255:
                raise ImagePolicyError(415, 'Şeffaf görseller desteklenmiyor. Fotoğrafın asıl dosyasını seçin.')
    return image


def create_app(engine_factory=E92Engine, *, inference_timeout=90.0, upload_timeout=30.0):
    slot = threading.BoundedSemaphore(1)
    state = {'engine': None}

    @asynccontextmanager
    async def lifespan(app):
        try:
            state['engine'] = await run_in_threadpool(engine_factory)
        except Exception:
            logger.exception('E92 initialization failed; no fallback')
        yield
        state['engine'] = None

    api = FastAPI(title='PixelProof local E92 research demo', lifespan=lifespan)
    origins = [part.strip() for part in os.environ.get('PIXELPROOF_CORS_ORIGINS',
                'http://localhost:3000,http://127.0.0.1:3000').split(',') if part.strip()]
    if '*' in origins:
        raise ValueError('Explicit local demo origins required')
    api.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST'],
                       allow_headers=['Content-Type'], allow_credentials=False)

    @api.get('/health')
    def health():
        return {'status': 'ready' if state['engine'] is not None else 'unavailable',
                'model_id': MODEL_ID, 'guard_id': GUARD_ID, 'artifact_sha256': CANDIDATE_SHA,
                'research_only': True, 'downloads_allowed': False}

    @api.post('/analyze', response_model=DemoResult)
    async def analyze(request: Request):
        # Acquiring before upload also bounds aggregate request memory. No queue.
        if not slot.acquire(blocking=False):
            raise HTTPException(429, 'Başka bir görsel inceleniyor. Biraz sonra yeniden deneyin.')
        worker_owns_slot = False
        try:
            raw_length = request.headers.get('content-length')
            if raw_length is not None:
                try:
                    length = int(raw_length)
                except ValueError:
                    raise HTTPException(400, 'Dosya boyutu doğrulanamadı.') from None
                if length < 0:
                    raise HTTPException(400, 'Dosya boyutu doğrulanamadı.')
                if length > DEFAULT_LIMITS.max_upload_bytes:
                    raise HTTPException(413, 'Dosya 12 MB sınırını aşıyor.')
            if request.headers.get('content-type', '').split(';')[0].lower() not in \
                    ('image/jpeg', 'image/png', 'image/webp', 'application/octet-stream'):
                raise HTTPException(415, 'JPG, PNG veya WEBP fotoğraf seçin.')

            async def read_bounded():
                body = bytearray()
                async for chunk in request.stream():
                    if len(body) + len(chunk) > DEFAULT_LIMITS.max_upload_bytes:
                        raise HTTPException(413, 'Dosya 12 MB sınırını aşıyor.')
                    body.extend(chunk)
                return bytes(body)

            try:
                raw = await asyncio.wait_for(read_bounded(), timeout=upload_timeout)
            except TimeoutError:
                raise HTTPException(408, 'Dosya aktarımı zamanında tamamlanamadı.') from None
            try:
                image = await run_in_threadpool(decode_demo, raw)
            except ImagePolicyError as exc:
                raise HTTPException(exc.status_code, exc.detail) from None
            del raw
            if min(image.size) < 224:
                return DemoResult(outcome='uncertain', reason='image_too_small', width=image.width, height=image.height)
            engine = state['engine']
            if engine is None:
                raise HTTPException(503, 'Model henüz hazır değil. Biraz sonra yeniden deneyin.')

            def infer():
                try:
                    outcome, reason = engine.analyze(image)
                    return DemoResult(outcome=outcome, reason=reason, width=image.width, height=image.height)
                finally:
                    slot.release()

            task = asyncio.create_task(run_in_threadpool(infer))
            worker_owns_slot = True
            # A response timeout cannot cancel a running GPU kernel. Keep the slot
            # until that worker actually finishes, and consume any late exception.
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=inference_timeout)
            except TimeoutError:
                raise HTTPException(504, 'İnceleme çok uzun sürdü. Biraz bekleyip yeniden deneyin.') from None
            except Exception:
                logger.exception('E92 inference failed; no result returned')
                raise HTTPException(503, 'İnceleme tamamlanamadı. Lütfen yeniden deneyin.') from None
        finally:
            if not worker_owns_slot:
                slot.release()

    return api


app = create_app()
