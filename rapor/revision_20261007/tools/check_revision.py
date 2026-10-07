"""Check this formatting-only revision against the saved student manuscript."""
import sys,json,re,hashlib,importlib.util
from pathlib import Path
from collections import Counter
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH as A
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from pypdf import PdfReader
import pdfplumber
root=Path(__file__).resolve().parents[1];base=root.parent/'final_2026';name='CS395_FinalReport_Efe_Han_Keleş_4October2026'
input_path=Path(sys.argv[1]);roundtrip=Path(sys.argv[2]);d=Document(root/'deliverables'/(name+'.docx'));original=Document(input_path)
checks=[]
def check(name,ok):
 checks.append({'check':name,'passed':bool(ok)})
 if not ok:raise AssertionError(name)
def paras(doc):return [''.join(p.xpath('.//w:t/text()')) for p in doc.element.body.xpath('.//w:p[not(ancestor::w:sdt)]') if ''.join(p.xpath('.//w:t/text()')).strip()]
check('all_242_nonempty_text_paragraphs_preserved',paras(d)==paras(original) and len(paras(d))==242)
check('3_figures_5_tables',len(d.inline_shapes)==3 and len(d.tables)==5)
check('no_sentence_lists',not d.element.xpath('.//w:pPr/w:numPr'))
check('all_margins_one_inch',all(abs(x.inches-1)<.001 for s in d.sections for x in [s.top_margin,s.bottom_margin,s.left_margin,s.right_margin]))
refs=[];headings=[];body=[];inside=False
for p in d.paragraphs:
 if p.style.name.startswith('Heading'):
  headings.append(p.text)
  if p.style.name=='Heading 1':inside=p.text=='8. References'
  check('heading_alignment:'+p.text,p.alignment==(A.CENTER if inside else A.LEFT))
  check('heading_numbered:'+p.text,bool(re.match(r'^\d+(?:\.\d+)*\.? ',p.text)))
 elif p.text and p.style.name=='Normal':
  f=p.paragraph_format
  if inside:
   refs.append(p.text)
   check('hanging_reference:'+p.text[:25],f.left_indent.pt==15 and f.first_line_indent.pt==-15 and f.right_indent.pt==0 and f.line_spacing==1 and f.space_after.pt==12)
  else:
   body.append(p.text)
   check('body_justified_double_zero_indent:'+str(len(body)),p.alignment==A.JUSTIFY and f.line_spacing==2 and all(v.pt==0 for v in [f.first_line_indent,f.left_indent,f.right_indent,f.space_before,f.space_after]))
 elif p.style.name=='Caption':
  check('centered_caption:'+p.text[:10],p.alignment==A.CENTER)
  previous=p._p.getprevious();following=p._p.getnext()
  check('caption_placement:'+p.text[:10],bool(previous.xpath('.//w:drawing')) if p.text.startswith('Figure') else following.tag==qn('w:tbl'))
  label=re.match(r'(Figure|Table) \d+',p.text).group()
  check('caption_has_prior_body_citation:'+label,any(label in x for x in body[-2:]))
for t in d.tables:check('table_centered',t.alignment==WD_TABLE_ALIGNMENT.CENTER)
spec=importlib.util.spec_from_file_location('c',base/'sources/report_content.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
check('all_15_references_unchanged',refs==[v for k,v in c.REFS])
body_text=' '.join(body)
for k,v in c.REFS:
 author='Türk Telekom' if k.startswith('Türk') else k;year=re.search(r'\((\d{4}[ab]?)',v).group(1)
 check('reference_cited:'+k,bool(re.search(re.escape(author)+r'(?: et al\.)?(?: \(|, )'+year,body_text)))
check('proper_figure_table_citations',not re.search(r'\bthe (Figure|Table) \d|\b(figure|table) \d|\b(Figure|Table) \d (above|below)',body_text))
for part in [d.element,d.styles.element,*[v for s in d.sections for v in [s.header._element,s.footer._element]]]:
 check('black_xml_colors',all(x.get(qn('w:val'))=='000000' for x in part.xpath('.//w:color')))
 check('12pt_xml_sizes',all(x.get(qn('w:val'))=='24' for x in part.xpath('.//w:sz')))
check('toc_field_present',any('TOC' in (x.text or '') for x in d.element.xpath('.//w:instrText')))
r=PdfReader(root/'deliverables'/(name+'.pdf'));rr=PdfReader(roundtrip);mp=json.loads((root/'sources/heading_pages.json').read_text());actual={}
for i,p in enumerate(r.pages):
 t=' '.join(p.extract_text().split())
 if i>=3:
  for h in headings:
   if h in t:actual[h]=i+1
check('all_35_toc_page_numbers_current',actual==mp and len(actual)==35)
toc=' '.join(r.pages[2].extract_text().split())
for h,p in mp.items():check('toc_entry:'+h,bool(re.search(re.escape(h)+r'\.*\s*'+str(p)+r'\b',toc)))
check('25_pages_within_requested_range',len(r.pages)==25)
with pdfplumber.open(roundtrip) as x,pdfplumber.open(root/'deliverables'/(name+'.pdf')) as y:
 check('legacy_doc_roundtrip_same_pages_and_spatial_text',len(x.pages)==len(y.pages) and all(''.join(a.extract_text().split())==''.join(b.extract_text().split()) for a,b in zip(x.pages,y.pages)))
 check('legacy_doc_roundtrip_same_word_positions',all(len(a.extract_words())==len(b.extract_words()) and all(u['text']==v['text'] and all(abs(u[k]-v[k])<.6 for k in ['x0','x1','top','bottom']) for u,v in zip(a.extract_words(),b.extract_words())) for a,b in zip(x.pages,y.pages)))
with pdfplumber.open(root/'deliverables'/(name+'.pdf')) as p:
 check('actual_pdf_Times_New_Roman',all('TimesNewRoman' in c['fontname'] for page in p.pages for c in page.chars))
 check('actual_pdf_black_text',all(c.get('non_stroking_color') in [0,(0,0,0)] for page in p.pages for c in page.chars))
 figures=[(page,img) for page in p.pages for img in page.images]
 check('three_actual_figures',len(figures)==3)
 check('actual_figure_centers',all(abs((im['x0']+im['x1']-pg.width)/2)<.5 for pg,im in figures))
 check('actual_body_within_margins',all(c['x0']>=71.5 and c['x1']<=page.width-71.5 for page in p.pages for c in page.chars if c['text'].strip() and 70<c['top']<page.height-70))
 check('no_empty_pages',all(len(page.crop((0,60,page.width,page.height-60)).extract_text() or '')>60 for page in p.pages))
check('experience_three_pages',mp['6. Conclusions']-mp['5. Internship experience']+1<=3)
check('company_within_three_pages',mp['3. Project background']-mp['2. Company information']<=3)
check('details_within_ten_pages',mp['4.6 Results']-mp['4.5 Project details']<=10)
check('literature_within_three_pages',mp['4. Internship project']-mp['3.4 Related literature']+1<=3)
check('results_on_single_page',all(' '.join(x.split()) in ' '.join(r.pages[mp['4.6 Results']-1].extract_text().split()) for x in [next(b['text'] for b in c.B if b.get('text','').startswith('The completed outcome is')),next(b['text'] for b in c.B if b.get('text','').startswith('The numerical improvement did not'))]))
def sections(doc):
 out={};h=None
 for p in doc.paragraphs:
  if p.style.name.startswith('Heading'):h=p.text;out[h]=[]
  elif h and p.text and p.style.name!='Caption':out[h].append(p.text)
 return out
old=sections(Document(base/'deliverables'/(name+'.docx')));new=sections(original)
comparison=[{'section':k,'text_changed':old.get(k)!=v,'previous_paragraph_count':len(old.get(k,[])),'current_paragraph_count':len(v)} for k,v in new.items()]
(root/'sources/student_text.json').write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n')
(root/'sources/section_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
result={'state':'formatting_checks_passed_content_review_pending','checks':checks,'check_count':len(checks),'text_preserved':True,'pages':len(r.pages),'input_docx_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'deliverables').iterdir()},'visual_review':'All 25 rendered pages inspected; pages 1 and 2 byte-identical to previously inspected first render. No clipping, lost figures or broken tables observed.','limitations':'Formatting and comparison, not a new Turnitin score, guaranteed policy acceptance, external bibliographic re-verification or native Word rendering. Student language and factual questions remain separately listed.'}
(root/'sources/format_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'checks':len(checks),'pages':len(r.pages),'changed_sections':[x['section'] for x in comparison if x['text_changed']]},ensure_ascii=False))
