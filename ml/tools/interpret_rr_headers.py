"""Bounded post-audit interpretation of E143 headers; no new image reads or fit."""
import json
from collections import Counter
from pathlib import Path
from pixelproof.header_declarations import declaration
from pixelproof.e92_demo import digest
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


def main():
    root=DATA_ROOT/'e143'; evidence=ML_ROOT.parent/'evidence'
    report=json.loads((evidence/'e143_rr_metadata.json').read_text())
    if digest(root/'private_header_records.json')!=report['private_records_sha256']:
        raise ValueError('Registered header output changed')
    rows=json.loads((root/'private_header_records.json').read_text())['rows']
    records=[]
    for row in rows:
        hint=declaration(row['fields'])
        if hint:
            records.append(dict(parent_id=row['parent_id'],label=row['label'],source=row['source'],
                declaration=hint,field_sha256=row['fields'][hint['field']]['sha256']))
    with (root/'private_header_declarations.json').open('x') as f:
        json.dump({'rows':records},f,indent=2,sort_keys=True);f.write('\n')
    summary=dict(state='RR_embedded_declarations_interpreted',metadata_record_sha256=report['private_records_sha256'],
        private_declarations_sha256=digest(root/'private_header_declarations.json'),
        declarations_by_class={str(y):dict(Counter(r['declaration']['value'] for r in records if r['label']==y)) for y in (0,1)},
        parents_with_AI_declaration=sum(r['label']==1 for r in records),
        remaining_RR_AI_without_parsed_declaration=report['by_class']['1']['parents']-sum(r['label']==1 for r in records),
        verified_generator_identities=0,downloads=0,new_image_reads=0,model_scores=0,
        code_sha256=digest(Path(__file__)),parser_sha256=digest(ML_ROOT/'src/pixelproof/header_declarations.py'),
        limits='Post-hoc interpretation of two exact observed field conventions, not blind discovery or certified generator labels. Metadata can be copied/edited; no version/checkpoint/base-family proof. Matplotlib/software/camera tags are not generator identities. Preserve original labels, source folds and unknowns; no automatic corpus/family merge or training admission.')
    with (evidence/'rr_header_declarations_20260916.json').open('x') as f:
        json.dump(summary,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
