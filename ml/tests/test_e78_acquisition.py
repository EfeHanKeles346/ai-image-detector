import pytest
from experiments import e78_acquisition as m


def test_only_pinned_public_dear_r_payload_admitted():
    metadata={'sha':m.REVISION,'gated':False,'private':False,'disabled':False,
              'siblings':[{'rfilename':'dear_r/model_best.pth','size':m.SIZE,'lfs':{'size':m.SIZE,'sha256':m.SHA}}]}
    assert m.select(metadata)['sha256']==m.SHA
    for changed in [metadata|{'sha':'main'},metadata|{'gated':True},metadata|{'siblings':[]},metadata|{'siblings':metadata['siblings']*2}]:
        with pytest.raises(ValueError):m.select(changed)


class Response:
    status_code=200
    headers={}
    def __init__(self,chunks):self.chunks=chunks;self.read=False
    def iter_content(self,n):self.read=True;return iter(self.chunks)


def test_wrong_headers_refused_before_body(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'resource_check',lambda deadline:None)
    for status,headers in [(206,{}),(200,{'Content-Length':'100'}),(200,{'Content-Encoding':'gzip'})]:
        response=Response([b'ab']);response.status_code=status;response.headers=headers
        with pytest.raises(ValueError):m.write_stream(response,tmp_path/'part',2,100)
        assert not response.read and not (tmp_path/'part').exists()


def test_stream_cap_truncation_and_exact_body(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'resource_check',lambda deadline:None)
    with pytest.raises(ValueError,match='cap'):m.write_stream(Response([b'ab',b'c']),tmp_path/'over',2,100)
    assert (tmp_path/'over').stat().st_size==2
    with pytest.raises(ValueError,match='incomplete'):m.write_stream(Response([b'a']),tmp_path/'short',2,100)
    assert m.write_stream(Response([b'a',b'b']),tmp_path/'okay',2,100)==2
    assert (tmp_path/'okay').read_bytes()==b'ab'
