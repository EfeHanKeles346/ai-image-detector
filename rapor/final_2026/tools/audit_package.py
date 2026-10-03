#!/usr/bin/env python3
"""Check the submitted package, never opening datasets or model weights."""
from pathlib import Path
import json, re, zipfile, importlib.util, hashlib, subprocess
from lxml import etree as E
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pypdf import PdfReader
import pdfplumber
from pdf_fonts import uses_embedded_times_new_roman
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1];OUT=ROOT/'deliverables'
spec=importlib.util.spec_from_file_location('c',ROOT/'sources/report_content.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
checks=[]
def check(name,ok):
 checks.append({'name':name,'passed':bool(ok)})
 if not ok:raise AssertionError(name)
name='CS395_FinalReport_Efe_Han_Keleş_29September2026'
d=Document(OUT/(name+'.docx'));r=PdfReader(OUT/(name+'.pdf'))
check('report_within_requested_20_to_25_pages',20<=len(r.pages)<=25)
check('abstract_at_most_250_words',len(c.ABSTRACT.split())<=250)
check('references_count_15',len(c.REFS)==15)
check('figures_3_tables_5',len(d.inline_shapes)==3 and len(d.tables)==5)
with pdfplumber.open(OUT/(name+'.pdf')) as rendered:
 figures=[(page,img) for page in rendered.pages for img in page.images]
 check('three_rendered_figures',len(figures)==3)
 for i,(page,img) in enumerate(figures,1):
  check('rendered_figure_centered:'+str(i),abs((img['x0']+img['x1']-page.width)/2)<.5)
  check('rendered_figure_within_text_width:'+str(i),img['x0']>=71.5 and img['x1']<=page.width-71.5)
for sec in d.sections:check('one_inch_margins',all(abs(x.inches-1)<.0001 for x in [sec.top_margin,sec.bottom_margin,sec.left_margin,sec.right_margin]))
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(OUT/(name+'.docx')) as z:
 for part in ['word/document.xml','word/styles.xml','word/header1.xml','word/footer1.xml']:
  tree=E.fromstring(z.read(part))
  check(part+':black_text',all(x.get('{'+ns['w']+'}val')=='000000' for x in tree.xpath('//w:color',namespaces=ns)))
 tree=E.fromstring(z.read('word/document.xml'))
 check('no_sentence_lists',not tree.xpath('//w:numPr',namespaces=ns))
 reference_texts={txt for _,txt in c.REFS}
 for paragraph in tree.xpath('//w:p[w:pPr/w:ind]',namespaces=ns):
  paragraph_text=''.join(paragraph.xpath('.//w:t/text()',namespaces=ns))
  if paragraph_text in reference_texts:continue
  check('non_reference_no_explicit_indentation',all(v=='0' for x in paragraph.xpath('./w:pPr/w:ind',namespaces=ns) for v in x.attrib.values()))
 check('toc_field_present',any('TOC' in (x.text or '') for x in tree.xpath('//w:instrText',namespaces=ns)))
 check('encoded_runs_12pt',all(x.get('{'+ns['w']+'}val')=='24' for x in tree.xpath('//w:sz',namespaces=ns)))
 for x in tree.xpath('//w:rFonts',namespaces=ns):check('encoded_run_Times_New_Roman',all(v=='Times New Roman' for v in x.attrib.values()))
headings=[b['text'] for b in c.B if b['type']=='heading'];m=json.loads((ROOT/'sources/heading_pages.json').read_text());actual={}
for i,page in enumerate(r.pages):
 text=' '.join(page.extract_text().split())
 if i>=3:
  for h in headings:
   if h in text:actual[h]=i+1
check('all_35_toc_pages_match_render',len(actual)==35 and actual==m)
toc=' '.join(' '.join(p.extract_text().split()) for p in r.pages[2:3])
for h in headings:check('toc_entry:'+h,bool(re.search(re.escape(h)+r'\.*\s*'+str(m[h])+r'\b',toc)))
for p in d.paragraphs:
 if p.style.name.startswith('Heading'):
  alignment=p.alignment if p.alignment is not None else p.style.paragraph_format.alignment
  expected=WD_ALIGN_PARAGRAPH.CENTER if p.text=='8. References' else WD_ALIGN_PARAGRAPH.LEFT
  check('numbered_heading_alignment:'+p.text,bool(re.match(r'^\d+(?:\.\d+)*\.? ',p.text)) and alignment==expected)
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
 reference=next(p for p in d.paragraphs if p.text==txt);fmt=reference.paragraph_format
 check('reference_five_space_hanging_indent:'+key,fmt.left_indent is not None and fmt.first_line_indent is not None and abs(fmt.left_indent.pt-15)<.01 and abs(fmt.first_line_indent.pt+15)<.01 and fmt.right_indent==0)
 check('reference_single_spacing_and_blank_line:'+key,fmt.line_spacing==1 and fmt.space_after.pt==12 and fmt.keep_together)
 author='Türk Telekom' if key.startswith('Türk') else key
 year=re.search(r'\((\d{4}[ab]?)',txt).group(1)
 check('reference_cited:'+key,bool(re.search(re.escape(author)+r'(?: et al\.)?(?: \(|, )'+year,body)))
 check('reference_has_full_entry:'+key,bool(re.search(r'Retrieved September (?:25|29), 2026, from https://',txt)) and bool(re.search(r'\(\d{4}[ab]?(?:, [A-Za-z]+ \d{1,2})?\)\.',txt)) and len(txt.split('Retrieved')[0].split())>=7)
# Page-span checks use the rendered body, not estimated word counts.
def section_span(heading):
 start=next(i for i,b in enumerate(c.B) if b.get('text')==heading)
 end=next(i for i in range(start+1,len(c.B)) if c.B[i]['type']=='heading')
 last=next(b['text'] for b in reversed(c.B[start+1:end]) if b['type']=='paragraph')
 suffix=' '.join(last.split()[-12:])
 pages=[i+1 for i,p in enumerate(r.pages) if suffix in ' '.join(p.extract_text().split())]
 if len(pages)!=1: raise AssertionError('Ambiguous section ending: '+heading)
 return pages[0]-m[heading]+1
check('company_at_most_3_pages',m['3. Project background']-m['2. Company information']<=3)
check('literature_at_most_3_pages',m['4. Internship project']-m['3.4 Related literature']<=3)
check('details_at_most_10_pages',m['4.6 Results']-m['4.5 Project details']<=10)
check('results_one_page',section_span('4.6 Results')==1)
check('experience_at_most_3_pages',m['6. Conclusions']-m['5. Internship experience']<=3)
check('conclusions_one_page',section_span('6. Conclusions')==1)
check('recommendations_one_page',section_span('7. Recommendations')==1)
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
policy=json.loads((REPO/'evidence/project_audit_20260916.json').read_text())['current_policy_counts']
check('paired_UI_REAL_139_clear_21_uncertain',policy['0']=={'no_clear_signal':139,'uncertain':21})
check('paired_UI_AI_159_alert_1_uncertain',policy['1']=={'ai_signal':159,'uncertain':1})
check('report_distinguishes_per_view_from_paired_UI','not the paired website outcomes in Section 4.5.6' in body)
check('model2_not_promoted',not json.loads((REPO/'evidence/e135_location_learning.json').read_text())['promotion_allowed'])
lineage=json.loads((REPO/'evidence/e117_coco_ancestry.json').read_text())
check('model2_506_of_512_protected_parent_overlap',lineage['author_parents']==512 and lineage['matched_original_parents']==506 and not lineage['training_allowed'])
masks=json.loads((REPO/'evidence/e109_diffseg_audit.json').read_text())
check('model2_diffseg_mask_strata',masks['decoded_pairs']==512 and masks['nondegenerate_pairs']==201 and masks['problems']=={'empty_edit_mask_origin_unresolved':87,'full_image_edit_mask':224} and not masks['training_allowed'])
pixels=json.loads((REPO/'evidence/e127_pixel_audit.json').read_text())
raw=pixels['matched_accepted_subset']['raw']
check('model2_e127_11_of_16_scope',pixels['parents_accounted']==16 and pixels['accepted_output_pairs']==11 and pixels['excluded_attempts']==5)
check('model2_e127_pixel_change_and_magnitude',round(raw['outside_changed_pixel_fraction']['mean']*100,2)==99.94 and round(raw['outside_mae']['mean']*255,2)==9.79)
check('model2_composite_preserves_background',pixels['matched_accepted_subset']['composite']['outside_mae']['max']==0)
check('model2_later_16_generation_gate',json.loads((REPO/'evidence/e129_context_replay.json').read_text())['comparison']['context_passed']==16)

a={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
for kind,count,table_count,chart_count in [('Presentation',12,4,1),('Digest',1,0,0)]:
 path=OUT/f'CS395_{kind}_Efe_Han_Keleş_29September2026.pptx'
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
notes=json.loads((ROOT/'sources/presentation_notes.json').read_text())
check('planned_timing_10_minutes',sum(x['seconds'] for x in notes)==600)
check('ten_spoken_slides_two_reference_slides',len(notes)==12 and all(x['seconds']>0 for x in notes[:10]) and all(x['seconds']==0 for x in notes[10:]))
files={}
for p in sorted(OUT.iterdir()):
 check('under_10MB:'+p.name,p.stat().st_size<10_000_000);files[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 if p.suffix=='.pdf':check('actual_pdf_fonts_embedded_Times_New_Roman:'+p.name,uses_embedded_times_new_roman(p))
check('seven_primary_files',len(files)==7)
placeholders=[match for p in d.paragraphs for match in re.findall(r'\[TO COMPLETE[^\]]*\]',p.text)]
result={'state':'pass_with_student_deferred_fields' if placeholders else 'pass','report_pages':len(r.pages),'report_headings':35,'figures':3,'tables':5,'references':len(c.REFS),'checks_passed':len(checks),'checks':checks,'files':files,'unresolved_fields':len(placeholders),'unresolved_placeholders':placeholders,'claim_boundary':'Formatting and selected evidence assertions; not exhaustive Markdown semantic review, scientific generalization, personal reflection verification, grade guarantee or native Microsoft Office testing.'}
(ROOT/'sources/package_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:result[k] for k in ['state','report_pages','checks_passed','unresolved_fields']},indent=2))
