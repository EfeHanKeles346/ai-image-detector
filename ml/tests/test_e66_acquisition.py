import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments import e66_acquisition as a


@pytest.mark.parametrize('status',[200,206])
def test_archive_resume_checks_range_and_handles_ignored_range(tmp_path,monkeypatch,status):
    body=b'checked archive body';archive=tmp_path/'archive.zip';part=archive.with_suffix('.zip.part')
    part.write_bytes(body[:5]);evidence=tmp_path/'evidence';evidence.mkdir()
    row={'url':'http://example.invalid/archive','bytes':len(body),'checksum_type':'md5',
         'checksum':hashlib.md5(body).hexdigest(),'body_sha1':hashlib.sha1(body).hexdigest()}
    contract=tmp_path/'contract.json';contract.write_text('{}')
    for k,v in {'ROOT':tmp_path,'ARCHIVE':archive,'EVIDENCE':evidence,'RECEIPT':tmp_path/'download.json','CONTRACT':contract}.items():
        monkeypatch.setattr(a,k,v)
    monkeypatch.setattr(a,'validate',lambda:{'row':row,'max_seconds':10})
    monkeypatch.setattr(Path,'is_mount',lambda self:True)
    monkeypatch.setattr(a.shutil,'disk_usage',lambda p:SimpleNamespace(free=10**12))
    class Response:
        status_code=status
        headers={'Content-Range':f'bytes 5-{len(body)-1}/{len(body)}'} if status==206 else {}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,*args):yield body[5:] if status==206 else body
    def get(url,**kwargs):
        assert kwargs['headers']=={'Range':'bytes=5-'}
        return Response()
    monkeypatch.setattr(a.requests,'get',get)
    result=a.download()
    assert archive.read_bytes()==body
    assert result['archive_sha256']==hashlib.sha256(body).hexdigest()
    assert not part.exists()
