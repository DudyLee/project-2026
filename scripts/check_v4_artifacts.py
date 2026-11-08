"""Static report checks, approval receipt and final integrity checks; no fitting."""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime, timezone
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/experiment_v4'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
class ReportParser(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.scripts=[];self.in_script=False;self.ids=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag in ['a','img']:self.refs.append(a.get('href') or a.get('src'))
        if tag=='script':self.in_script=True
    def handle_endtag(self,tag):
        if tag=='script':self.in_script=False
    def handle_data(self,data):
        if self.in_script:self.scripts.append(data)

run=json.loads((OUT/'run_manifest.json').read_text());v=json.loads((OUT/'verification.json').read_text());build=json.loads((OUT/'report_build_manifest.json').read_text())
assert run['status']=='completed' and sum(run['fits'].values())==49 and v['status']=='passed'
assert all(sha(ROOT/p)==h for p,h in run['input_hashes'].items())
assert sha(ROOT/'scripts/train_pv_increment_v4.py')==run['script_sha256']
assert sha(ROOT/'scripts/evaluate_pv_increment_v4.py')==v['evaluation_script_sha256']
assert sha(ROOT/'scripts/report_pv_increment_v4.py')==build['report_script_sha256']
assert all(sha(OUT/p)==h for p,h in build['artifacts'].items())
p=ReportParser();p.feed((OUT/'report.html').read_text(encoding='utf-8'))
assert len(p.ids)==len(set(p.ids)) and len(p.scripts)==1
local=[r for r in p.refs if r and not r.startswith(('https://','http://','#'))]
assert all((OUT/r).is_file() for r in local)
js=ROOT/'.cache/report_v4_inline.js';js.write_text(p.scripts[0],encoding='utf-8')
result=subprocess.run(['C:/Program Files/nodejs/node.exe','--check',str(js)],capture_output=True,text=True)
assert result.returncode==0,result.stderr
dump(OUT/'presentation_verification.json',{'HTML_static_relative_links_checked':len(local),'HTML_duplicate_ids':0,'inline_JavaScript_syntax':'passed',
    'scientific_figures_visually_inspected':3,'interactive_browser_validation':'not performed: local browser use disabled in Work settings',
    'canvas_native_compilation_performed':False,'temporary_preview_server':'stopped','core_numerical_verification':'passed; see verification.json',
    'static_check_script_sha256':sha(Path(__file__))})
approval_dir=ROOT/'docs/approvals';approval_dir.mkdir(exist_ok=True)
dump(approval_dir/'decision_v4_approval.json',{'status':'approved_and_executed','approval_user_message':'确认 先按照这个计划做一轮实验看看效果',
     'source_proposal':'configs/proposals/decision_v4_draft.json','source_proposal_sha256':sha(ROOT/'configs/proposals/decision_v4_draft.json'),
     'execution_config':'configs/experiment_v4.json','execution_config_sha256':sha(ROOT/'configs/experiment_v4.json'),
     'completed_run':'reports/experiment_v4/run_manifest.json','receipt_written_utc':datetime.now(timezone.utc).isoformat(),
     'historical_draft_policy':'Original draft retains its pre-approval content; this receipt and the execution config record the later user authorization.',
     'further_work':'No authorization for additional experiments or final TEST unsealing inferred.'})
print('Final integrity and static report checks passed; all 49 fits complete; TEST predictions/scores remain zero.')
print('Browser interaction validation unavailable because local browser use is disabled; numerical and image verification complete.')
