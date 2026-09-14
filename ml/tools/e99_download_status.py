"""Update the DATASETS acquisition ledger from verified per-file receipts, not file counts."""
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.e65_acquisition import digest, read
from experiments.e99_acquisition import ROOT, validate
from pixelproof.project_paths import ML_ROOT


def main():
    c = validate(); binding = digest(ROOT/'acquisition_contract.json')
    counts = []
    for package in c['packages']:
        completed = 0; body_bytes = 0
        for row in package['rows']:
            image = ROOT/'images'/package['sensor']/Path(row['filename']).name
            receipt = image.with_suffix('.receipt.json')
            if not receipt.exists():
                continue
            try:
                r = read(receipt)
            except json.JSONDecodeError:
                continue  # Concurrent writer has not completed this small receipt yet.
            if r['contract_sha256'] != binding or r['member'] != row['filename'] or \
                    r['bytes'] != row['bytes'] or r['crc32'] != row['crc32'] or \
                    not image.exists() or image.stat().st_size != row['bytes']:
                raise ValueError('Receipt no longer matches selected body')
            completed += 1; body_bytes += r['bytes']
        counts.append({'sensor': package['sensor'], 'requested': len(package['rows']),
                       'verified_files': completed, 'verified_body_bytes': body_bytes})
    complete = (ROOT/'download.json').exists()
    if complete and digest(ROOT/'download.json') != read(ML_ROOT.parent/'evidence/e99_download.json')['receipt_sha256']:
        raise ValueError('Final receipt mismatch')
    total = sum(r['verified_files'] for r in counts)
    if complete and total != c['selected']:
        raise ValueError('Final receipt has missing bodies')
    state = 'COMPLETE — quarantined, admission separate' if complete else 'INCOMPLETE — transfer/checks not finished'
    lines = ['<!-- E99-LIVE-START -->', '### E99 current download snapshot', '',
             f"Updated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. **{state}**.", '',
             '| Sensor | Requested files | Verified files | Verified body bytes |',
             '|---|---:|---:|---:|']
    for r in counts:
        lines.append(f"|{r['sensor']}|{r['requested']}|{r['verified_files']}|{r['verified_body_bytes']:,}|")
    lines += ['', f"Total: {total}/{c['selected']} files; {sum(r['verified_body_bytes'] for r in counts):,} verified body bytes.",
              'Counts use exact selected receipt paths; AppleDouble sidecars/partial files do not count.',
              'This snapshot verifies receipt binding and current file size; original CRC/SHA checks are',
              'performed by the acquisition worker and are rechecked during admission. Image bodies stay',
              'under `$PIXELPROOF_DATA_ROOT/e99/images`. No new model score or quality claim.',
              '<!-- E99-LIVE-END -->']
    path = ML_ROOT.parent/'DATASETS.md'; text = path.read_text(); block = '\n'.join(lines)
    if '<!-- E99-LIVE-START -->' in text:
        start = text.index('<!-- E99-LIVE-START -->'); end = text.index('<!-- E99-LIVE-END -->', start)+len('<!-- E99-LIVE-END -->')
        text = text[:start]+block+text[end:]
    else:
        text += '\n\n'+block+'\n'
    path.write_text(text)
    print(json.dumps({'state': state, 'counts': counts, 'total': total}))


if __name__ == '__main__':
    main()
