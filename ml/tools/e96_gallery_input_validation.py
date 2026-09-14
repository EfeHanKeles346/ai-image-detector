"""Private-pixel admission parity and targeted HTTP replay, not a new ML benchmark."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image, ImageOps
from pixelproof.project_paths import ML_ROOT, WORK_ROOT
from pixelproof.e92_demo import digest
from pixelproof.demo_image_input import decode_photo
from pixelproof.research_serve import decode_demo
from pixelproof.image_input import ImagePolicyError

ROOT = WORK_ROOT/'e95_owner_gallery'
OUT = ML_ROOT.parent/'evidence'


def read(p): return json.loads(p.read_text())

def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')


def parity():
    rows=read(ROOT/'contract.json')['files']; old_ok=new_ok=0;selected=[]
    files=[Path(__file__),ML_ROOT/'src/pixelproof/demo_image_input.py',ML_ROOT/'src/pixelproof/internship_serve.py']
    inputs={str(p):digest(p) for p in files}
    for row in rows:
        p=Path(row['path'])
        if digest(p)!=row['sha256']:raise ValueError('Gallery identity changed')
        raw=p.read_bytes();new=decode_photo(raw)
        with Image.open(p) as opened: expected=ImageOps.exif_transpose(opened).convert('RGB')
        np.testing.assert_array_equal(np.asarray(new),np.asarray(expected));new_ok+=1
        try:
            old=decode_demo(raw)
            np.testing.assert_array_equal(np.asarray(new),np.asarray(old));old_ok+=1
        except ImagePolicyError:
            selected.append(row)
    chosen=sorted(selected,key=lambda r:(r['sha256'],r['name']))[:3]
    write(ROOT/'e96_http_selection.json',{'rows':chosen})
    result={'state':'E96_decoder_parity_passed','files':len(rows),'old_accepted':old_ok,'new_accepted':new_ok,
            'newly_accepted':len(selected),'native_rgb_pixel_changes':0,'input_code_sha256':inputs,
            'selection_sha256':digest(ROOT/'e96_http_selection.json'),
            'limits':'Admission only; no model scores, fitting, raw image output or threshold changes. Three new24MP HTTP cases selected by file SHA before scoring.'}
    write(OUT/'e96_gallery_input_parity.json',result)
    return result


def http():
    parity_result=read(OUT/'e96_gallery_input_parity.json')
    for p,h in parity_result['input_code_sha256'].items():
        if digest(Path(p))!=h:raise ValueError('E96 source changed')
    if digest(ROOT/'e96_http_selection.json')!=parity_result['selection_sha256']:raise ValueError('Selection changed')
    rows=read(ROOT/'e96_http_selection.json')['rows']; previous=read(ROOT/'scores.json')['rows']
    if digest(ROOT/'scores.json')!=read(OUT/'e95_gallery_scores_receipt.json')['scores_sha256']:raise ValueError('E95 scores changed')
    expected={r['sha256']:r['display'] for r in previous}
    ai=[r for r in read(WORK_ROOT/'e94_display_rows.json') if r['label']==1]
    for outcome,warning in sorted({(r['outcome'],r['review_required']) for r in ai}):
        row=min((r for r in ai if (r['outcome'],r['review_required'])==(outcome,warning)),key=lambda r:r['sha256'])
        rows.append(row);expected[row['sha256']]=row
    cases=[]
    for row in rows:
        p=Path(row['path']);raw=p.read_bytes()
        if digest(p)!=row['sha256']:raise ValueError('HTTP image changed')
        with Image.open(p) as im:kind='image/png' if im.format=='PNG' else 'image/jpeg'
        request=urllib.request.Request('http://127.0.0.1:8800/analyze',data=raw,headers={'Content-Type':kind},method='POST')
        start=time.monotonic()
        with urllib.request.urlopen(request,timeout=100) as response:actual=json.load(response)
        for key in ['outcome','guard_outcome','reason','review_required','display_policy']:
            if actual[key]!=expected[row['sha256']][key]:raise ValueError('Current display changed in HTTP replay')
        cases.append({'width':actual['width'],'height':actual['height'],'seconds':time.monotonic()-start,
                      'outcome':actual['outcome'],'review_required':actual['review_required']})
    result={'state':'E96_HTTP_replay_passed','cases':cases,'private_gallery_cases':3,'consumed_AI_cases':len(rows)-3,
            'newly_missed_AI':0,'limits':'Targeted integration parity, not independent accuracy/AI retention proof. No private filenames or pixels exported.'}
    write(OUT/'e96_gallery_http.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['parity','http'])
    print(json.dumps({'parity':parity,'http':http}[p.parse_args().stage](),indent=2))
