"""Read-only content inventory and redacted credential scan before the first push."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import os
import re

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.conda','.cache','.venv','venv','__pycache__','.pytest_cache','.mypy_cache','node_modules','.idea','.vscode','.aws','.codex','.agents'}
GENERATED={'docs/repository_snapshot.json','docs/repository_preflight.json'}
TEXT={'.py','.ps1','.json','.md','.txt','.yml','.yaml','.html','.tsx','.csv','.js','.toml','.ini','.cfg'}
PATTERNS={
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'openai_key':re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b'),
    'aws_access_key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'slack_token':re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b'),
    'credential_url':re.compile(r'https?://[^\s/:@]+:[^\s/@]{6,}@'),
    'assigned_secret':re.compile(r'''(?im)["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)["']?\s*[:=]\s*["']([^"'\r\n]{12,})["']'''),
}


def main():
    rows=[];findings=[];groups=defaultdict(lambda:{'files':0,'bytes':0});scanned=0
    for base,dirs,files in os.walk(ROOT):
        dirs[:]=sorted(d for d in dirs if d not in SKIP)
        for name in sorted(files):
            path=Path(base)/name;rel=path.relative_to(ROOT).as_posix()
            if rel in GENERATED or path.suffix in {'.pyc','.pyo'}:continue
            if name.startswith('.env') or path.suffix in {'.pem','.key','.p12','.pfx'} or name in {'credentials.json','secrets.json'}:
                findings.append({'path':rel,'type':'credential_filename'});continue
            content=path.read_bytes();size=len(content)
            assert size<100*2**20, f'File requires separate large-file handling: {rel}'
            rows.append({'path':rel,'bytes':size,'sha256':hashlib.sha256(content).hexdigest()})
            group=rel.split('/')[0];groups[group]['files']+=1;groups[group]['bytes']+=size
            if path.suffix.lower() in TEXT:
                scanned+=1;text=content.decode('utf-8-sig',errors='replace')
                for category,pattern in PATTERNS.items():
                    for match in pattern.finditer(text):
                        # Only paths/line numbers are reported. Never print candidate credentials.
                        findings.append({'path':rel,'type':category,'line':text.count('\n',0,match.start())+1})
    audit={'date':'2026-11-08','status':'passed' if not findings else 'review_required','files_in_inventory':len(rows),
           'text_files_scanned':scanned,'content_bytes':sum(x['bytes'] for x in rows),'groups':dict(groups),
           'findings':findings,'scan_scope':'Common credential patterns and filenames; candidate values never logged.',
           'excluded_directories':sorted(SKIP),'destination':'Intended private GitHub repository DudyLee/project-2026',
           'test_handling':'Storage backup only. No numeric parsing, fitting, prediction or scoring of sealed TEST rows.'}
    (ROOT/'docs/repository_preflight.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if findings:
        print(json.dumps({'status':'review_required','findings':findings},ensure_ascii=False,indent=2))
        raise SystemExit(1)
    snapshot={'date':'2026-11-08','scope':'Initial private repository snapshot, excluding environment/cache/credentials',
              'excluded_self_records':sorted(GENERATED),'files':rows}
    (ROOT/'docs/repository_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','files':len(rows),'MiB':round(audit['content_bytes']/2**20,2),
                      'text_files_scanned':scanned,'credential_findings':0},ensure_ascii=False))


if __name__=='__main__':main()
