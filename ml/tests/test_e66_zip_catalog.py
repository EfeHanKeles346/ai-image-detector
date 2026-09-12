import io
import zipfile
import pytest
from experiments import e66_zip_catalog as c


def test_range_reader_supports_zip_metadata_without_full_transfer(monkeypatch):
    source=io.BytesIO()
    with zipfile.ZipFile(source,'w') as z:z.writestr('scene/image.png',b'x'*1000000)
    raw=source.getvalue();requested=[]
    class Response:
        status_code=206
        def __init__(self,a,b):self.a=a;self.b=b;self.headers={'Content-Range':f'bytes {a}-{b}/{len(raw)}'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,*args):yield raw[self.a:self.b+1]
    def get(url,**kwargs):
        a,b=map(int,kwargs['headers']['Range'].removeprefix('bytes=').split('-'))
        requested.append((a,b));return Response(a,b)
    monkeypatch.setattr(c.requests,'get',get)
    reader=c.RangeReader('http://example.invalid/zip',len(raw),budget=100000)
    with zipfile.ZipFile(reader) as z:assert z.namelist()==['scene/image.png']
    assert reader.used<100000
    assert all(a>900000 for a,b in requested)
    reader.seek(0)
    with pytest.raises(ValueError,match='budget'):reader.read()
