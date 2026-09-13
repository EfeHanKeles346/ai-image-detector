import ast
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import pytest
from experiments import e75_features as m


def test_crop_transport_bytes_equal_historical_e54_without_import_side_effects():
    tree=ast.parse(Path(m.__file__).with_name('e54_data.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='crop_views')
    scope={'Image':Image,'ImageOps':ImageOps,'BytesIO':BytesIO,'np':np,'dino':m.dino}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'historical_pure_crop_views','exec'),scope)
    rng=np.random.default_rng(75)
    im=Image.fromarray(rng.integers(0,256,(317,2177,3),dtype=np.uint8));exif=Image.Exif();exif[274]=6
    stream=BytesIO();im.save(stream,format='JPEG',quality=91,exif=exif);raw=stream.getvalue()
    parents={}
    for i in range(100):
        p=f'E75_parity:{i}';parents.setdefault(m.dino.assigned_transport(p),p)
    assert len(parents)==4
    for parent in parents.values():
        actual=m.crop_views(raw,parent)
        assert actual.shape==(3,3,224,224,3) and actual.dtype==np.uint8
        assert np.array_equal(actual,scope['crop_views'](raw,parent))


def row():return {'parent_id':'MIDD:test:1','sha256':'f'*64,'role':'TRAIN','source':'MIDD:test','label':0,'training_allowed':True}


def test_unadmitted_and_duplicate_population_rejected():
    m.validate_population([row()])
    for change in [{'role':'DEVELOPMENT'},{'label':1},{'source':'SIDD:camera'},{'training_allowed':False}]:
        with pytest.raises(ValueError,match='inadmissible'):m.validate_population([row()|change])
    with pytest.raises(ValueError,match='unique'):m.validate_population([row(),row()])
    with pytest.raises(ValueError,match='nonempty'):m.validate_population([])


def test_resume_chunks_bind_parent_source_and_feature_body(tmp_path):
    dino=np.zeros((3,3072),dtype=np.float32);clip=np.zeros((3,1536),dtype=np.float32)
    arrays={'dino':dino,'clip':clip,'dino_sha256':m.array_sha(dino),'clip_sha256':m.array_sha(clip),
            'binding':'contract','parent_id':row()['parent_id'],'source_sha256':row()['sha256']}
    path=tmp_path/'chunk.npz';np.savez_compressed(path,**arrays)
    assert np.array_equal(m.read_chunk(path,'contract',row())[0],dino)
    with pytest.raises(ValueError,match='binding'):m.read_chunk(path,'different',row())
    with pytest.raises(ValueError,match='source'):m.read_chunk(path,'contract',row()|{'sha256':'a'*64})
    arrays['clip']=clip.copy();arrays['clip'][0,0]=1;np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='body'):m.read_chunk(path,'contract',row())
