"""SIDD member/decode/group audit and limited paired-label DEV admission, without scores."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import fcntl
import json
from pathlib import Path,PurePosixPath
import re
import zipfile

from experiments.e65_acquisition import digest,read,write_once
from experiments.e65_audit import fingerprint,cross_role_matches,REFERENCES,VERSION
from experiments.e66_acquisition import ROOT,ARCHIVE,CONTRACT as ACQUISITION,validate as validate_acquisition
from pixelproof.project_paths import DATA_ROOT,ML_ROOT

EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'sidd_audit_contract.json'
REPORT=ROOT/'sidd_audit.json'
MANIFEST=ROOT/'development_manifest.json'
INSTANCE=re.compile(r'^(\d{4})_(\d{3})_(GP|IP|S6|N6|G4)_(\d{5})_(\d{5})_(\d{4})_([LNH])$')


def select_noisy(entries):
    instances=defaultdict(dict)
    for r in entries:
        p=PurePosixPath(r['filename'])
        if p.is_absolute() or '..' in p.parts or '\\' in r['filename']:
            raise ValueError('unsafe ZIP member path')
        if r['directory']:continue
        if p.name in ['_ReadMe.txt','Scene_Instances.txt']:
            if r['bytes']>100000:raise ValueError('metadata exceeds size budget')
            continue
        if len(p.parts)!=4 or p.parts[:2]!=('SIDD_Small_sRGB_Only','Data'):
            raise ValueError('unexpected archive member')
        m=INSTANCE.fullmatch(p.parts[2]);kind=re.fullmatch(r'(NOISY|GT)_SRGB_(\d{3})\.PNG',p.name)
        if not m or not kind or not 0<r['bytes']<100*1024**2:raise ValueError('invalid SIDD member metadata')
        identity,scene,camera,iso,shutter,temp,bright=m.groups()
        if kind[1] in instances[identity]:raise ValueError('duplicate SIDD companion kind')
        instances[identity][kind[1]]=r|{'parent_id':'SIDD:'+identity,'scene_group':'SIDD:scene:'+scene,
            'camera':camera,'iso':int(iso),'shutter_denominator':int(shutter),'temperature_kelvin':int(temp),
            'brightness':bright,'frame':kind[2],'source':'SIDD:'+camera,'label':0,'role':'DEVELOPMENT',
            'condition':'publisher_noisy_srgb','training_allowed':False}
    selected=[]
    for pair in instances.values():
        if set(pair)!= {'NOISY','GT'} or pair['NOISY']['frame']!=pair['GT']['frame'] or \
                pair['NOISY']['scene_group']!=pair['GT']['scene_group']:
            raise ValueError('missing/mismatched SIDD companion')
        selected.append(pair['NOISY'])
    return sorted(selected,key=lambda r:r['parent_id'])


def propagate_scene_failures(rows,failed):
    scenes={r['scene_group'] for r in rows if r['parent_id'] in set(failed)}
    return {r['parent_id'] for r in rows if r['scene_group'] in scenes}


def freeze():
    validate_acquisition()
    directory=ROOT/'sidd_zip_directory.json';catalog=read(directory)
    if catalog['acquisition_contract_sha256']!=digest(ACQUISITION):raise ValueError('ZIP directory binding differs')
    rows=select_noisy(catalog['entries'])
    if len(rows)!=160 or len({r['scene_group'] for r in rows})!=10:raise ValueError('SIDD expected scene coverage differs')
    ai=ROOT/'ai_admission.json'
    if digest(ai)!=read(EVIDENCE/'e66_ai_admission.json')['report_sha256'] or read(ai)['admitted']!=160:
        raise ValueError('complete AI DEV admission required')
    inputs={str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('e65_audit.py'),
        Path(__file__).with_name('e65_acquisition.py'),Path(__file__).with_name('e66_acquisition.py'),
        directory,ACQUISITION,ai,ROOT/'ai_contract.json']}
    for p,s in REFERENCES.items():
        if digest(DATA_ROOT/p)!=s:raise ValueError('protected snapshot changed')
        inputs[str(DATA_ROOT/p)]=s
    e65=DATA_ROOT/'e65/audit.json'
    if digest(e65)!=read(EVIDENCE/'e65_audit.json')['report_sha256']:raise ValueError('E65 audit changed')
    inputs[str(e65)]=digest(e65)
    c={'state':'E66_SIDD_grouped_DEV_audit_registered','inputs':inputs,'rows':rows,'max_pixels':100000000,
       'policy':'Full publisher archive checksum receipt required; verify archive SHA256 and exact complete '
                'ZIP directory against pre-pixel metadata. Materialize only160 NOISY PNG members to flat '
                'owned paths; streaming size/CRC checked. GT remains unextracted companion evidence. '
                'Canonical body/RGB or dHash<=4+pHash<=4 cross-reference matches quarantine entire underlying '
                'SIDD scene. Internal same-scene matches retained as dependent; cross-scene matches quarantine '
                'both scenes. No replacements or score-based exclusions. Final DEV manifest requires160 '
                'SIDD observations+160 admitted AI with no shared body/id or known TRAIN role.',
       'training_allowed':False,'model_scoring_allowed':False,'independent_final_allowed':False,
       'limitations':'Only10 real scenes across5 older phones; camera and scene dependence must be reported. '
                    'SIDD renderings and two previously seen AI families support a limited DEV rejection '
                    'screen, not modern-source final/generalization proof. AI prompt dependencies unknown.'}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e66_sidd_audit_contract.json',{k:v for k,v in c.items() if k!='rows'}|
        {'selected_instances':len(rows),'scenes':10,'cameras':dict(Counter(r['camera'] for r in rows)),
         'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'selected':len(rows),'scene_groups':10}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e66_sidd_audit_contract.json')['contract_sha256']:raise ValueError('SIDD audit contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('SIDD bound input changed')
    return c


def audit():
    c=validate();validate_acquisition()
    if REPORT.exists() or MANIFEST.exists():raise FileExistsError('SIDD audit already complete')
    receipt=read(ROOT/'download.json')
    if digest(ROOT/'download.json')!=read(EVIDENCE/'e66_download.json')['receipt_sha256'] or \
            receipt['contract_sha256']!=digest(ACQUISITION) or digest(ARCHIVE)!=receipt['archive_sha256']:
        raise ValueError('verified SIDD archive identity changed')
    (ROOT/'sidd').mkdir(exist_ok=True);facts=[];failed=[]
    with zipfile.ZipFile(ARCHIVE) as archive:
        actual=[{'filename':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,
               'crc32':f'{i.CRC:08x}','compression':i.compress_type,'directory':i.is_dir(),
               'header_offset':i.header_offset} for i in archive.infolist()]
        if actual!=read(ROOT/'sidd_zip_directory.json')['entries']:raise ValueError('full ZIP directory differs')
        for i,r in enumerate(c['rows']):
            try:
                info=archive.getinfo(r['filename'])
                if info.flag_bits&1 or (info.external_attr>>16)&0o170000==0o120000:
                    raise ValueError('encrypted or symlink image member')
                raw=archive.read(info)  # zipfile validates CRC; pinned directory bounds decompressed size.
                if len(raw)!=r['bytes']:raise ValueError('extracted member size differs')
                f=fingerprint(raw,max_pixels=c['max_pixels'])
                if min(f['width'],f['height'])<224:raise ValueError('SIDD member below224px')
                p=ROOT/'sidd'/(r['parent_id'].split(':')[1]+'.png')
                if p.exists():
                    if digest(p)!=f['sha256']:raise ValueError('previously materialized member differs')
                else:
                    temp=p.with_suffix('.png.part');temp.write_bytes(raw);temp.replace(p)
                facts.append(r|f|{'path':str(p)})
            except (ValueError,OSError,zipfile.BadZipFile) as error:
                failed.append({'parent_id':r['parent_id'],'reason':str(error)})
            if (i+1)%10==0:print(json.dumps({'SIDD_processed':i+1,'decoded':len(facts),'failed':len(failed)}),flush=True)
    refs=[]
    for p in REFERENCES:refs.extend(read(DATA_ROOT/p)['records'])
    refs.extend(read(DATA_ROOT/'e65/audit.json')['records'])
    if any(r['version']!=VERSION for r in refs):raise ValueError('reference convention differs')
    matches=cross_role_matches(facts,refs);internal=[m for m in cross_role_matches(facts,facts) if m['train_parent']<m['cal_parent']]
    by_id={r['parent_id']:r for r in facts}
    cross_scene=[m for m in internal if by_id[m['train_parent']]['scene_group']!=by_id[m['cal_parent']]['scene_group']]
    rejected=({r['parent_id'] for r in failed}|{r['train_parent'] for r in matches}|
        {p for m in cross_scene for p in [m['train_parent'],m['cal_parent']]})
    bad=propagate_scene_failures(c['rows'],rejected);admitted=[r for r in facts if r['parent_id'] not in bad]
    ai=read(ROOT/'ai_admission.json')['records']
    cross_label=cross_role_matches(admitted,ai)
    complete=(len(admitted)==160 and len(ai)==160 and not cross_label
              and len({r['sha256'] for r in admitted+ai})==320
              and len({r['parent_id'] for r in admitted+ai})==320)
    result={'state':'E66_SIDD_DEV_admitted' if complete else 'E66_SIDD_DEV_incomplete',
        'contract_sha256':digest(CONTRACT),'records':admitted,'decode_failures':failed,
        'cross_matches':matches,'internal_matches':internal,'cross_scene_matches':cross_scene,
        'cross_label_matches':cross_label,'quarantined_parents':sorted(bad),'admitted':len(admitted),
        'scene_groups':len({r['scene_group'] for r in admitted}),'reference_records':len(refs),
        'camera_counts':dict(Counter(r['camera'] for r in admitted)),'model_scores_created':0,
        'training_allowed':False,'balanced_development_admitted':complete,'limitations':c['limitations']}
    write_once(REPORT,result)
    summary={k:v for k,v in result.items() if k not in ['records','cross_matches','internal_matches','cross_scene_matches','cross_label_matches']}
    summary.update(report_sha256=digest(REPORT),cross_match_observations=len(matches),
        internal_match_pairs=len(internal),cross_scene_match_pairs=len(cross_scene),cross_label_match_pairs=len(cross_label))
    write_once(EVIDENCE/'e66_sidd_admission.json',summary)
    if complete:
        rows=admitted+ai
        if len({r['sha256'] for r in rows})!=len(rows) or len({r['parent_id'] for r in rows})!=len(rows):
            raise ValueError('duplicate DEV identity')
        manifest={'state':'E66_limited_balanced_DEV_frozen_unscored','rows':rows,'sidd_audit_sha256':digest(REPORT),
            'ai_audit_sha256':digest(ROOT/'ai_admission.json'),'training_allowed':False,'model_scores_created':0,
            'class_counts':{'0':160,'1':160},'independent_final_admitted':False,'limitations':c['limitations']}
        write_once(MANIFEST,manifest)
        write_once(EVIDENCE/'e66_development.json',{k:v for k,v in manifest.items() if k!='rows'}|
                   {'manifest_sha256':digest(MANIFEST),'parents_or_observations':len(rows)})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','audit'))
    with (ROOT/'sidd_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'audit':audit}[parser.parse_args().stage](),indent=2))
