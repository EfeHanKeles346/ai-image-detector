import io
import time
import zipfile
import zlib
import pytest
from experiments import e72_acquisition as m


def entry(name):
    return {'filename':name,'directory':False,'bytes':3,'compressed_bytes':3,'compression':0,
            'crc32':'352441c2','header_offset':0}


def test_selection_excludes_test_denoised_and_unsafe_members_and_is_order_independent():
    sensor='Sony_IMX258'
    rows=[entry(f'{sensor}/training_set/original/{n}.jpg') for n in range(140)]
    rows += [entry(f'{sensor}/test_set/original/1.jpg'),entry(f'{sensor}/training_set/denoised/1.jpg'),
             entry(f'{sensor}/training_set/original/../1.jpg'),entry('._Sony_IMX258/training_set/original/1.jpg')]
    catalog={'sensor':sensor,'url':m.BASE_URL+sensor+'.zip','entries':rows}
    selected=m.select(catalog)
    assert len(selected)==128
    assert all('/training_set/original/' in r['filename'] and '..' not in r['filename'] for r in selected)
    assert selected==m.select(catalog | {'entries':rows[::-1]})
    with pytest.raises(ValueError,match='duplicate'):
        m.select(catalog | {'entries':rows+[rows[0]]})
    with pytest.raises(ValueError,match='insufficient'):
        m.select(catalog | {'entries':rows[:127]})


def test_zip_member_integrity_detects_metadata_and_body_change():
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,'w') as z:z.writestr('Sony_IMX258/training_set/original/1.jpg',b'abc')
    with zipfile.ZipFile(io.BytesIO(raw.getvalue())) as z:
        info=z.infolist()[0];row=entry(info.filename)
        m.member_identity(info,row)
        assert len(m.verify_body(z.read(info),row))==64
        with pytest.raises(ValueError,match='identity'):m.member_identity(info,row | {'header_offset':9})
        with pytest.raises(ValueError,match='CRC'):m.verify_body(b'abd',row)
        with pytest.raises(ValueError,match='length'):m.verify_body(b'ab',row)


class Response:
    def __init__(self,status=206,headers=None,body=b'abc'):
        self.status_code=status;self.headers=headers or {'Content-Range':'bytes 0-2/9','ETag':'"fixed"'}
        self.body=body;self.consumed=False
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):pass
    def iter_content(self,size):self.consumed=True;yield self.body


def reader():
    return m.PinnedRanges('https://example.test/archive',{'bytes':9,'etag':'"fixed"'},3,time.monotonic()+30)


def test_exact_ranges_send_if_match_and_enforce_budget(monkeypatch):
    calls=[];response=Response()
    def get(url,**kwargs):calls.append(kwargs);return response
    monkeypatch.setattr(m.requests,'get',get)
    r=reader();assert r.read(3)==b'abc';assert r.tell()==3 and r.used==3
    assert calls[0]['headers']['If-Match']=='"fixed"'
    assert calls[0]['headers']['Range']=='bytes=0-2'
    with pytest.raises(ValueError,match='transfer'):r.read(1)
    assert len(calls)==1


@pytest.mark.parametrize('response,reason',[
    (Response(status=200),'exact range'),
    (Response(headers={'Content-Range':'bytes 1-3/9'}),'exact range'),
    (Response(headers={'Content-Range':'bytes 0-2/9','ETag':'"changed"'}),'ETag'),
    (Response(body=b'ab'),'truncated'),(Response(body=b'abcd'),'overlong')])
def test_range_failures_do_not_advance_or_accept_bulk(monkeypatch,response,reason):
    monkeypatch.setattr(m.requests,'get',lambda *args,**kwargs:response)
    r=reader()
    with pytest.raises(ValueError,match=reason):r.read(3)
    assert r.tell()==0 and r.used==0
    if reason in ('exact range','ETag'):assert not response.consumed
