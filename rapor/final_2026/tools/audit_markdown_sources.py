#!/usr/bin/env python3
"""Inventory every line of the pre-review tracked Markdown snapshot.

This deterministic structural scan is not an automated semantic fact checker.
Semantic review and its completion ledger are reported separately.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'rapor/final_2026/sources/markdown_scan.json'
SNAPSHOT = '16e03d832c57cea2cea8806807660e1389fe9f1e'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def scan():
    paths = sorted(p for p in git('ls-tree', '-r', '--name-only', SNAPSHOT).decode().splitlines()
                   if p.lower().endswith('.md'))
    files = []
    for path in paths:
        raw = git('show', f'{SNAPSHOT}:{path}')
        lines = raw.decode('utf-8').splitlines(keepends=True)
        sections = []
        start = 1
        title = '(preamble)'
        fence = None
        for n, line in enumerate(lines, 1):
            fence_match = re.match(r'^\s*(`{3,}|~{3,})', line)
            if fence_match:
                delimiter = fence_match.group(1)
                if fence is None:
                    fence = delimiter[0]
                elif delimiter[0] == fence:
                    fence = None
                continue
            if fence is None and re.match(r'^#{1,6}\s+', line):
                if n > start:
                    sections.append((start, n - 1, title))
                start, title = n, line.strip()
        if lines:
            sections.append((start, len(lines), title))
        covered = []
        rows = []
        for first, last, title in sections:
            text = ''.join(lines[first-1:last])
            covered.extend(range(first, last+1))
            refs = sorted(set(re.findall(r'\bevidence/[A-Za-z0-9_.\-/]+\.json\b', text)))
            rows.append({
                'start_line': first, 'end_line': last, 'heading': title,
                'sha256': hashlib.sha256(text.encode()).hexdigest(),
                'experiment_mentions': sorted(set(re.findall(r'\bE\d{1,3}[a-zA-Z]?\b', text))),
                'receipt_mentions': refs,
                'correction_marker_lines': [first+i for i, line in enumerate(lines[first-1:last])
                    if re.search(r'correct|supersed|errat|invalid|leak|revers|historical', line, re.I)],
            })
        assert covered == list(range(1, len(lines)+1)), path
        files.append({'path': path, 'bytes': len(raw), 'lines': len(lines),
                      'sha256': hashlib.sha256(raw).hexdigest(), 'sections': rows})
    result = {
        'snapshot': SNAPSHOT,
        'method': 'Full UTF-8 byte read; contiguous non-overlapping section and line inventory; all receipt, experiment and correction-marker references scanned.',
        'semantic_review_performed_by_scanner': False,
        'semantic_review_ledger': 'markdown_read_progress.json',
        'boundary': 'A complete structural scan is not a complete line-by-line semantic review. This file does not certify every historical claim, external reference or experiment.',
        'scope': 'All tracked project Markdown at the pre-review commit, including old and current report sources. Ignored datasets, dependencies and third-party runtime trees are outside the project-document scope.',
        'file_count': len(files), 'line_count': sum(f['lines'] for f in files),
        'byte_count': sum(f['bytes'] for f in files),
        'section_count': sum(len(f['sections']) for f in files), 'files': files,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: result[k] for k in ['snapshot','file_count','line_count','byte_count','section_count','semantic_review_performed_by_scanner']}, indent=2))

if __name__ == '__main__':
    scan()
