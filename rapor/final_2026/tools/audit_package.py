#!/usr/bin/env python3
"""Check the submitted package, never opening datasets or model weights."""
from pathlib import Path
import json, re, zipfile, importlib.util, hashlib, subprocess
from lxml import etree as E
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1];OUT=ROOT/'deliverables'
spec=importlib.util.spec_from_file_location('c',ROOT/'sources/report_content.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
checks=[]
def check(name,ok):
 checks.append({'name':name,'passed':bool(ok)})
 if not ok:raise AssertionError(name)
name='CS395_FinalReport_EfeHan_Keles_25September2026'
d=Document(OUT/(name+'.docx'));r=PdfReader(OUT/(name+'.pdf'))
check('report_30_pages',len(r.pages)==30)
check('abstract_at_most_250_words',len(c.ABSTRACT.split())<=250)
check('references_14_including_10_scholarly',len(c.REFS)==14)
check('figures_3_tables_6',len(d.inline_shapes)==3 and len(d.tables)==6)
for sec in d.sections:check('one_inch_margins',all(abs(x.inches-1)<.0001 for x in [sec.top_margin,sec.bottom_margin,sec.left_margin,sec.right_margin]))
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(OUT/(name+'.docx')) as z:
 for part in ['word/document.xml','word/styles.xml','word/header1.xml','word/footer1.xml']:
  tree=E.fromstring(z.read(part))
  check(part+':black_text',all(x.get('{'+ns['w']+'}val')=='000000' for x in tree.xpath('//w:color',namespaces=ns)))
 tree=E.fromstring(z.read('word/document.xml'))
 check('no_sentence_lists',not tree.xpath('//w:numPr',namespaces=ns))
 check('no_explicit_nonzero_indentation',all(v=='0' for x in tree.xpath('//w:ind',namespaces=ns) for k,v in x.attrib.items()))
 check('toc_field_present',any('TOC' in (x.text or '') for x in tree.xpath('//w:instrText',namespaces=ns)))
 check('encoded_runs_12pt',all(x.get('{'+ns['w']+'}val')=='24' for x in tree.xpath('//w:sz',namespaces=ns)))
 for x in tree.xpath('//w:rFonts',namespaces=ns):check('encoded_run_Times_New_Roman',all(v=='Times New Roman' for v in x.attrib.values()))
headings=[b['text'] for b in c.B if b['type']=='heading'];m=json.loads((ROOT/'sources/heading_pages.json').read_text());actual={}
for i,page in enumerate(r.pages):
 text=' '.join(page.extract_text().split())
 if i>=4:
  for h in headings:
   if h in text:actual[h]=i+1
check('all_35_toc_pages_match_render',len(actual)==35 and actual==m)
toc=' '.join(' '.join(p.extract_text().split()) for p in r.pages[2:4])
for h in headings:check('toc_entry:'+h,bool(re.search(re.escape(h)+r'\.*\s*'+str(m[h])+r'\b',toc)))
for p in d.paragraphs:
 if p.style.name.startswith('Heading'):
  check('numbered_left_heading',bool(re.match(r'^\d+(?:\.\d+)*\.? ',p.text)) and p.style.paragraph_format.alignment==WD_ALIGN_PARAGRAPH.LEFT)
for idx,b in enumerate(c.B):
 if b['type']=='paragraph':
  ps=[p for p in d.paragraphs if p.text==b['text']];check('body_double_justified',len(ps)==1 and ps[0].paragraph_format.line_spacing==2 and (ps[0].alignment or ps[0].style.paragraph_format.alignment)==WD_ALIGN_PARAGRAPH.JUSTIFY)
 if b['type'] in ['figure','table']:
  label=b['type'].capitalize()+' '+str(b['number'])
  check('nearby_preceding_citation:'+label,any(label in x.get('text','') for x in c.B[max(0,idx-2):idx] if x['type']=='paragraph'))
  p=next(p for p in d.paragraphs if p.text.startswith(label+'. '));check('centered_caption:'+label,p.alignment==WD_ALIGN_PARAGRAPH.CENTER)
  sib=p._p.getprevious() if b['type']=='figure' else p._p.getnext()
  check('caption_placement:'+label,bool(sib.xpath('.//w:drawing')) if b['type']=='figure' else sib.tag.endswith('}tbl'))
for t in d.tables:check('centered_table',t.alignment==WD_TABLE_ALIGNMENT.CENTER)
body=' '.join(b.get('text','') for b in c.B if b['type']=='paragraph')
check('no_relative_or_lowercase_figure_citations',not re.search(r'\bthe (Figure|Table) \d|\b(figure|table) \d|\b(Figure|Table) \d (above|below)',body))
for key,txt in c.REFS:
 author='Türk Telekom' if key.startswith('Türk') else key
 check('reference_cited:'+key,author in body)
 check('reference_has_full_entry:'+key,'Retrieved September 25, 2026, from https://' in txt and bool(re.search(r'\((?:\d{4}[ab]?|n\.d\.)\)\.',txt)) and len(txt.split('Retrieved')[0].split())>=7)
# Page-span checks use the rendered body, not estimated word counts.
check('company_at_most_3_pages',m['3. Project background']-m['2. Company information']<=3)
check('literature_at_most_3_pages',m['4. Internship project']-m['3.4 Related literature']<=3)
check('details_at_most_10_pages',m['4.6 Results']-m['4.5 Project details']<=10)
check('results_one_page',m['5. Internship experience']-m['4.6 Results']==1)
check('experience_at_most_3_pages',m['6. Conclusions']-m['5. Internship experience']<=3)
check('conclusions_one_page',m['7. Recommendations']-m['6. Conclusions']==1)
check('recommendations_one_page',m['8. References']-m['7. Recommendations']==1)
e=json.loads((REPO/'evidence/e92_development.json').read_text())
for cond,fp in [('publisher_original',0),('social_q75',14)]:
 b=e['reports'][cond]['new']['binary_metrics'];check('E92_confusion:'+cond,b['confusion']['tp']==159 and b['confusion']['fn']==1 and b['confusion']['fp']==fp)
 check('E92_numeric_gates:'+cond,e['checks'][cond]['absolute_gates_passed'])
check('E92_full_acceptance_failed',not e['passes_limited_dev_screen'] and not e['checks']['publisher_original']['zero_new_ai_misses'] and not e['independent_final_passed'])
rr=json.loads((REPO/'evidence/e42_rr_result.json').read_text())
check('E42_RR_distinct_parents_and_views',rr['by_condition']['original']['metrics']['counts']['total']==16953 and sum(v['metrics']['counts']['total'] for v in rr['by_condition'].values())==50858)
check('E42_RR_original_false_alerts',rr['by_condition']['original']['metrics']['confusion']['fp']==2052)
final=json.loads((REPO/'evidence/e49_final_result.json').read_text())
check('E49_comprehensive_failed_11_of_20',final['parent_count']==2000 and final['observation_count']==4000 and final['gate']=={'passed':False,'passed_checks':11,'total_checks':20})
for cond,fp,tp in [('publisher_original',391,943),('social_q75',490,955)]:
 check('E49_confusion:'+cond,final['conditions'][cond]['binary_metrics']['confusion']['fp']==fp and final['conditions'][cond]['binary_metrics']['confusion']['tp']==tp)
later=json.loads((REPO/'evidence/e102_development.json').read_text())
check('E102_social_false_alerts_12',later['reports']['social_q75']['new']['binary_metrics']['confusion']['fp']==12)
check('E102_full_acceptance_failed',not later['passes_limited_dev_screen'] and not later['independent_final_passed'])
check('model2_not_promoted',not json.loads((REPO/'evidence/e135_location_learning.json').read_text())['promotion_allowed'])
a={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
for kind,count,table_count,chart_count in [('Presentation',15,5,1),('Digest',1,0,0)]:
 path=OUT/f'CS395_{kind}_EfeHan_Keles_25September2026.pptx'
 with zipfile.ZipFile(path) as z:
  parts=[n for n in z.namelist() if re.fullmatch('ppt/slides/slide[0-9]+.xml',n)]
  check(kind+':slide_count',len(parts)==count)
  tables=0;charts=0
  for part in parts:
   t=E.fromstring(z.read(part));tables+=len(t.xpath('//a:tbl',namespaces=a));charts+=len(t.xpath('//*[local-name()="chart"]'))
   check(kind+':black_run_colors',all(x.get('val')=='000000' for x in t.xpath('//a:rPr/a:solidFill/a:srgbClr',namespaces=a)))
   check(kind+':font',all(x.get('typeface')=='Times New Roman' for x in t.xpath('//a:latin',namespaces=a)))
  check(kind+':native_tables_charts',tables==table_count and charts==chart_count)
  check(kind+':speaker_notes',len([n for n in z.namelist() if re.fullmatch('ppt/notesSlides/notesSlide[0-9]+.xml',n)])==count)
 check(kind+':pdf_pages',len(PdfReader(path.with_suffix('.pdf')).pages)==count)
notes=json.loads((ROOT/'sources/presentation_notes.json').read_text());check('planned_timing_10_to_15_minutes',600<=sum(x['seconds'] for x in notes)<=900)
files={}
for p in sorted(OUT.iterdir()):
 check('under_10MB:'+p.name,p.stat().st_size<10_000_000);files[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
check('seven_primary_files',len(files)==7)
placeholders=[{'heading_context':b.get('type'),'text':b.get('text')} for b in c.B if '[TO COMPLETE' in b.get('text','')]
result={'state':'pass_with_student_deferred_fields','report_pages':len(r.pages),'report_headings':35,'figures':3,'tables':6,'references':14,'checks_passed':len(checks),'checks':checks,'files':files,'unresolved_fields':len(placeholders)+2,'claim_boundary':'Formatting and selected evidence assertions; not exhaustive Markdown semantic review, scientific generalization, personal reflection verification, grade guarantee or native Microsoft Office testing.'}
(ROOT/'sources/package_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:result[k] for k in ['state','report_pages','checks_passed','unresolved_fields']},indent=2))
