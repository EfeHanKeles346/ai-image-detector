import asyncio
import threading
import httpx
import pytest
from PIL import Image
import pixelproof.internship_serve as serving
from pixelproof.primary_demo_policy import primary_result


@pytest.mark.parametrize('exit_mode', ['cancel', 'timeout'])
def test_cancelled_decode_keeps_single_worker_slot(monkeypatch, exit_mode):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = []

    def decode(_raw):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            finished.set()
        return Image.new('RGB', (224, 224))

    class Engine:
        def analyze(self, image):
            return primary_result([.9, .9], [0., 0.])

    monkeypatch.setattr(serving, 'decode_photo', decode)

    async def exercise():
        app = serving.create_app(Engine, inference_timeout=.03 if exit_mode == 'timeout' else 90.)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                first = asyncio.create_task(client.post('/analyze', content=b'synthetic', headers={'content-type': 'image/png'}))
                try:
                    for _ in range(200):
                        if entered.is_set():
                            break
                        await asyncio.sleep(.005)
                    assert entered.is_set()
                    if exit_mode == 'cancel':
                        first.cancel()
                        await asyncio.gather(first, return_exceptions=True)
                    else:
                        assert (await first).status_code == 504
                    second = await client.post('/analyze', content=b'synthetic', headers={'content-type': 'image/png'})
                    assert second.status_code == 429, 'Cancelled request must not release a running decoder slot'
                    assert len(calls) == 1
                finally:
                    release.set()
                    await asyncio.gather(first, return_exceptions=True)
                    for _ in range(200):
                        if finished.is_set():
                            break
                        await asyncio.sleep(.005)
                for _ in range(200):
                    response = await client.post('/analyze', content=b'synthetic', headers={'content-type': 'image/png'})
                    if response.status_code != 429:
                        break
                    await asyncio.sleep(.005)
                assert response.status_code == 200

    asyncio.run(exercise())
