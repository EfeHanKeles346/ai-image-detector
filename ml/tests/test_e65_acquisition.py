import hashlib
import time

import pytest
from experiments import e65_acquisition as a


def test_wifd_excludes_reference_and_collapses_repeated_settings():
    rows = [{'type': 'blob', 'path': f'dataset/camera/sdr_image/camera_sdr_image_ISO{iso}_1_100_5.0_{seq}.jpg',
             'size': 10, 'sha': 'a'*40} for iso in (100, 400, 1600) for seq in (1, 2)]
    rows.append({'type': 'blob', 'path': 'dataset/camera/reference/x.jpg', 'size': 1, 'sha': 'b'*40})
    result = a.choose_wifd({'sha': a.REVISION, 'truncated': False, 'tree': rows})
    assert len(result) == 2
    assert {r['iso'] for r in result} == {100, 1600}
    assert all(not r['training_allowed'] for r in result)
    assert a.choose_wifd({'sha': a.REVISION, 'truncated': False, 'tree': list(reversed(rows))}) == result


def test_incomplete_remote_inventory_rejected():
    with pytest.raises(ValueError, match='complete pinned'):
        a.choose_wifd({'sha': a.REVISION, 'truncated': True, 'tree': []})


def test_git_blob_hash_is_not_plain_content_sha1(tmp_path):
    path = tmp_path/'image'; body=b'image payload'; path.write_bytes(body)
    row={'bytes':len(body),'checksum_type':'git_blob_sha1',
         'checksum':hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest()}
    assert a.verify_file(path,row) == hashlib.sha256(body).hexdigest()
    row['checksum']=hashlib.sha1(body).hexdigest()
    with pytest.raises(ValueError,match='checksum'):
        a.verify_file(path,row)


def test_ignored_http_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    body=b'complete verified file'; path=tmp_path/'image.jpg'
    path.with_suffix('.jpg.part').write_bytes(body[:5])
    class Response:
        status_code=200
        headers={}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def raise_for_status(self): pass
        def iter_content(self,*args): yield body
    def get(url, **kwargs):
        assert kwargs['headers']=={'Range':'bytes=5-'}
        return Response()
    monkeypatch.setattr(a.requests,'get',get)
    row={'path':str(path),'url':'https://example.invalid/image','bytes':len(body),
         'checksum_type':'md5','checksum':hashlib.md5(body).hexdigest()}
    result=a.fetch_one(row,time.monotonic()+10)
    assert path.read_bytes()==body
    assert result['sha256']==hashlib.sha256(body).hexdigest()


def test_rawnind_excludes_restricted_and_non_cc0_and_official_test_scenes():
    names = ['Bayer_good_GT_ISO100_sha1=a.arw','Bayer_good_ISO6400_sha1=b.arw']
    file_rows=[{'restricted':False,'dataFile':{'filename':name,'id':i,'filesize':1,
                'checksum':{'type':'MD5','value':'0'*32}}} for i,name in enumerate(names)]
    scene={'test_reserve':False,'clean_images':[{'filename':names[0],'sha1':'a'}],
           'noisy_images':[{'filename':names[1],'sha1':'b'}]}
    metadata={'data':{'latestVersion':{'files':file_rows}}}
    dataset={'Bayer':{'good':scene},'X-Trans':{}}
    # An extended/unknown ISO string cannot silently be coerced to a numeric ISO.
    unknown='Bayer_good_ISOHi 3200_sha1=c.nef'
    scene['noisy_images'].append({'filename':unknown,'sha1':'c'})
    file_rows.append({'restricted':False,'dataFile':{'filename':unknown,'id':99,'filesize':1,
                     'checksum':{'type':'MD5','value':'0'*32}}})
    assert len(a.choose_rawnind(metadata,dataset,'good: CC0, by author',{'test_reserve':[]}))==2
    for rights,reserves in [('good: research only',[]),('good: CC0, by author',['good'])]:
        assert not a.choose_rawnind(metadata,dataset,rights,{'test_reserve':reserves})
    file_rows[0]['restricted']=True
    assert not a.choose_rawnind(metadata,dataset,'good: CC0, by author',{'test_reserve':[]})
