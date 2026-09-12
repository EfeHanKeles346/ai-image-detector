from io import BytesIO
import numpy as np
from PIL import Image
from experiments.e65_audit import fingerprint, cross_role_matches, propagate_quarantine


def row(parent, dh=0, ph=0, sha=None, pixel=None):
    return dict(parent_id=parent, canonical_dhash=f'{dh:016x}', phash63=f'{ph:016x}',
                sha256=sha or parent, pixel_sha256=pixel or parent, label=0, condition='original')


def test_perceptual_both_distances_required_but_exact_takes_precedence():
    left = [row('new')]
    right = [row('boundary',15,15), row('dh_too_far',31,0), row('ph_too_far',0,31),
             row('exact',2**64-1,2**63-1,pixel='new')]
    assert {r['cal_parent'] for r in cross_role_matches(left,right)} == {'boundary','exact'}


def test_raw_pair_quarantine_does_not_invent_wifd_scene_groups():
    rows=[dict(parent_id=p,source=s,scene_group=g) for p,s,g in
          [('low','RawNIND','A'),('high','RawNIND','A'),('other','RawNIND','B'),
           ('jpeg1','WIFD','unknown'),('jpeg2','WIFD','unknown')]]
    assert propagate_quarantine(rows,{'low','jpeg1'}) == {'low','high','jpeg1'}


def test_equal_pixels_different_encoding_match():
    image=Image.fromarray(np.random.default_rng(65).integers(0,256,(32,32,3),dtype=np.uint8))
    arrays=[]
    for level in [0,9]:
        stream=BytesIO();image.save(stream,format='PNG',compress_level=level)
        arrays.append(fingerprint(stream.getvalue()))
    assert arrays[0]['sha256'] != arrays[1]['sha256']
    assert arrays[0]['pixel_sha256'] == arrays[1]['pixel_sha256']
    assert arrays[0]['canonical_dhash'] == arrays[1]['canonical_dhash']
