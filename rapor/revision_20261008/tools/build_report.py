"""Insert the supplied manuscript verbatim; only layout/approved numbering changes."""
from pathlib import Path
from copy import deepcopy
import re,json,hashlib
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.shared import Pt,Inches,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH as A, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.enum.table import WD_TABLE_ALIGNMENT
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'revision_20261007/deliverables/CS395_FinalReport_Efe_Han_Keleş_4October2026.docx'
OUT=ROOT/'deliverables/CS395_FinalReport_Efe_Han_Keleş_4October2026.docx'
raw=(ROOT/'sources/student_input.txt').read_text()
old=Document(BASE);d=Document(BASE)
# Preserve the previous report's cover, relationships, images, reference runs and tables.
refs=[];inrefs=False;figs={};tables={}
for p in old.paragraphs:
 if p.style.name=='Heading 1':inrefs=p.text=='8. References'
 elif inrefs and p.text:refs.append(deepcopy(p._p))
 if p.style.name=='Caption':
  m=re.match(r'(Figure|Table) (\d+)',p.text)
  if m:
   k,n=m.group(1),int(m.group(2))
   if k=='Figure':figs[n]=[deepcopy(p._p.getprevious()),deepcopy(p._p)]
   else:tables[n]=[deepcopy(p._p),deepcopy(p._p.getnext())]
# Extract source sections without editing spelling or punctuation.
sections=[];current=None;buf=[];meta=[]
def flush():
 if current is not None:current['raw']='\n'.join(buf).strip();buf.clear()
for line in raw.splitlines():
 s=line.strip()
 if re.match(r'^\d+(?:\.\d+)*\.?\s*[A-Za-z]',s):
  flush();num=re.match(r'^(\d+(?:\.\d+)*)(?:\.(?!\d))?',s).group(1)
  current={'old':num,'heading':s};sections.append(current)
 elif s=='(abstact 3.2)':meta.append(s)
 else:buf.append(line)
flush()
by={s['old']:s for s in sections}
# Authorized structure change: criteria precede results, then shift later subsection IDs.
order=[s['old'] for s in sections if s['old'] not in ('9','9.1')]
order.insert(order.index('4.5.5'),'9.1')
num_map={'9.1':'4.5.5',**{f'4.5.{i}':f'4.5.{i+1}' for i in range(5,9)}}
tab_map={3:4,4:5,5:3}
simple_headings={'4.5.1': 'First models and early tests', '4.5.2': 'Finding and correcting errors', '4.5.3': 'Preparing the images', '4.5.4': 'How E92 works', '4.5.5': 'How we checked model performance', '4.5.6': 'E92 results and remaining problems', '4.5.7': 'How the web demo makes decisions', '4.5.8': 'Further tests and unsuccessful changes', '4.5.9': 'Detecting edited areas with Model2'}
page_map=json.loads((ROOT/'sources/heading_pages.json').read_text()) if (ROOT/'sources/heading_pages.json').exists() else {}
# Body rebuild starts at abstract; cover source paragraphs preserved.
body=d.element.body
start=next(p._p for p in d.paragraphs if p.text=='Abstract')
idx=list(body).index(start)
for el in list(body)[idx:]:
 if el.tag!=qn('w:sectPr'):body.remove(el)
def addp(text='',style='Normal'):
 p=d.add_paragraph(style=style);p.add_run(text);return p
def appendxml(el):body.insert(len(body)-1,deepcopy(el))
def num_text(t):
 return re.sub(r'\b([Tt]able) ([345])\b',lambda m:m[1]+' '+str(tab_map[int(m[2])]),t)
def display(t):
 # Markdown link syntax is formatting, not manuscript prose; keep exact displayed label.
 return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1',t)
addp('Abstract','Title').paragraph_format.page_break_before=True
addp('[TO COMPLETE]').runs[0].bold=True
addp('Table of Contents','Title').paragraph_format.page_break_before=True
# Cached, editable Word TOC. Values are refreshed deterministically from rendered pages.
sdt=OxmlElement('w:sdt');content=OxmlElement('w:sdtContent');sdt.append(content)
headers=[]
for key in order:
 s=by[key];h=s['heading']
 if key in num_map:h=re.sub(r'^'+re.escape(key),num_map[key],h, count=1)
 if num_map.get(key,key) in simple_headings:h=num_map.get(key,key)+' '+simple_headings[num_map.get(key,key)]
 headers.append(h)
for i,h in enumerate(headers):
 p=OxmlElement('w:p');pr=OxmlElement('w:pPr');p.append(pr)
 jc=OxmlElement('w:jc');jc.set(qn('w:val'),'left');pr.append(jc)
 tabs=OxmlElement('w:tabs');tab=OxmlElement('w:tab');tab.set(qn('w:val'),'right');tab.set(qn('w:leader'),'dot');tab.set(qn('w:pos'),'9026');tabs.append(tab);pr.append(tabs)
 def r(tag,text=None,attrs=None):
  rr=OxmlElement('w:r');e=OxmlElement('w:'+tag)
  if text is not None:e.text=text
  for k,v in (attrs or {}).items():e.set(qn('w:'+k),v)
  rr.append(e);p.append(rr)
 if i==0:
  r('fldChar',attrs={'fldCharType':'begin'});r('instrText',' TOC \\o "1-3" \\h \\z \\u ');r('fldChar',attrs={'fldCharType':'separate'})
 r('t',h);r('tab');r('t',str(page_map.get(h,1)))
 if i==len(headers)-1:r('fldChar',attrs={'fldCharType':'end'})
 content.append(p)
appendxml(sdt)
expected=[];placements=[]
def asset(kind,n):
 elems=figs[n] if kind=='Figure' else tables[n]
 for el in elems:
  e=deepcopy(el)
  if kind=='Table' and e.tag==qn('w:p'):
   for t in e.xpath('.//w:t'):t.text=num_text(t.text or '')
  appendxml(e)
 placements.append((kind,n))
for key,h in zip(order,headers):
 depth=num_map.get(key,key).count('.')+1
 p=addp(h,'Heading '+str(depth));p.paragraph_format.page_break_before=key in ('1','4.6','8')
 if key=='8':
  for ref in refs:appendxml(ref)
  continue
 text=by[key]['raw']
 paras=[x.strip() for x in re.split(r'\n\s*\n',text) if x.strip()]
 # Literature pasted as one run; separate studies without changing any non-whitespace character.
 if key=='3.4':
  starts=['Wang et al. (2020)','Ojha et al. (2023)','CLIP could','DINOv2 is','According to Grommelt','DEAR uses','SIDD is','The SID dataset','High score','Guo et al. (2017) explain','Dwork et al. (2015)','Adobe Content Credentials']
  for st in starts:text=text.replace(st,'\n\n'+st,1)
  paras=[x.strip() for x in re.split(r'\n\s*\n',text) if x.strip()]
 for i,t in enumerate(paras):
  t=num_text(display(t));expected.append({'section':h,'text':t});p=addp(t)
  if key=='2' and 'Figure 1' in t:asset('Figure',1)
  if key=='4.3' and 'Table 1' in t:asset('Table',1)
  if key=='4.5.3' and 'Table 2' in t:asset('Table',2)
  if key=='4.5.4' and 'Figure 2' in t:asset('Figure',2)
  if key=='4.5.5' and 'Figure 3' in t:asset('Figure',3)
  if key=='4.6' and 'Table 5' in t:asset('Table',4)
  if key=='9.1' and 'table 3' in t.lower():asset('Table',5)
 if key=='4.5.8':
  addp('[TO COMPLETE]').runs[0].bold=True
  asset('Table',3) # Existing evidence preserved; student prose and citation pending.
# Typography: professor rules override general skill/template recommendations.
for sec in d.sections:
 sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Inches(1)
for s in d.styles:
 if s.type in (1,2):s.font.name='Times New Roman';s.font.size=Pt(12);s.font.color.rgb=RGBColor(0,0,0)
refs_mode=False
for p in d.paragraphs:
 f=p.paragraph_format;t=p.text
 f.left_indent=f.right_indent=f.first_line_indent=Pt(0);f.space_before=f.space_after=Pt(0);f.line_spacing=2;f.widow_control=True
 p.alignment=A.JUSTIFY
 if t=='[TO COMPLETE]':f.keep_with_next=True
 if p.style.name.startswith('Heading'):
  refs_mode=t=='8 REFERENCES' if p.style.name=='Heading 1' else refs_mode
  p.alignment=A.CENTER if refs_mode else A.LEFT
  f.space_before=Pt(12);f.space_after=Pt(6);f.keep_with_next=True;f.keep_together=True
 elif p.style.name=='Title':
  p.alignment=A.LEFT;f.keep_with_next=True
  if t.startswith('PixelProof'):f.space_before=Pt(54);f.space_after=Pt(36)
 elif p.style.name=='Caption':
  p.alignment=A.CENTER;f.keep_together=True;f.keep_with_next=t.startswith('Table')
 elif p._p.xpath('.//w:drawing'):
  p.alignment=A.CENTER;f.line_spacing=1;f.keep_with_next=True
 elif refs_mode and t:
  f.left_indent=Pt(15);f.first_line_indent=Pt(-15);f.line_spacing=1;f.space_after=Pt(12);f.keep_together=True
for pxml in d.element.xpath('.//w:sdt//w:p'):
 p=Paragraph(pxml,d._body);p.paragraph_format.line_spacing=1.1;p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0)
for table in d.tables:
 table.alignment=WD_TABLE_ALIGNMENT.CENTER
 for row in table.rows:
  for cell in row.cells:
   for p in cell.paragraphs:
    p.alignment=A.JUSTIFY;f=p.paragraph_format;f.line_spacing=2;f.space_before=f.space_after=Pt(0);f.left_indent=f.right_indent=f.first_line_indent=Pt(0)
for part in [d.element,d.styles.element,*[x for s in d.sections for x in (s.header._element,s.footer._element)]]:
 for run in part.xpath('.//w:r'):
  rp=run.find(qn('w:rPr'))
  if rp is None:rp=OxmlElement('w:rPr');run.insert(0,rp)
  for tag in ['color','rFonts','sz','szCs']:
   el=rp.find(qn('w:'+tag))
   if el is None:el=OxmlElement('w:'+tag);rp.append(el)
   el.attrib.clear()
   if tag=='color':el.set(qn('w:val'),'000000')
   elif tag=='rFonts':
    for a in ['ascii','hAnsi','eastAsia','cs']:el.set(qn('w:'+a),'Times New Roman')
   else:el.set(qn('w:val'),'24')
# Preserve all words and punctuation; only paragraph whitespace/link syntax/approved table numbers differ.
canon=lambda t:re.sub(r'\s+','',t)
for key in order:
 if key=='8':continue
 src=num_text(display(by[key]['raw']))
 dst=''.join(x['text'] for x in expected if x['section']==headers[order.index(key)])
 assert canon(src)==canon(dst),key
OUT.parent.mkdir(parents=True,exist_ok=True);d.save(OUT)
(ROOT/'sources/expected_prose.json').write_text(json.dumps(expected,ensure_ascii=False,indent=2)+'\n')
(ROOT/'sources/structure.json').write_text(json.dumps({'headings':headers,'section_number_changes':num_map,'table_number_changes':tab_map,'nonreport_note_excluded':meta,'pending':['Abstract','4.5.9 Model2 (previously 4.5.8)'],'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),'prose_paragraphs':len(expected)},ensure_ascii=False,indent=2)+'\n')
print(OUT);print('paragraphs',len(expected),'tables',len(d.tables),'figures',len(d.inline_shapes))
