#!/usr/bin/env python3
"""Build the report and data-backed figures using bundled document dependencies."""
from pathlib import Path
import importlib.util, json, hashlib, re, sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from pdf2image import convert_from_path
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parents[1]
spec=importlib.util.spec_from_file_location('content',ROOT/'sources/report_content.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
BUILD=Path('/private/tmp/pixelproof-internship-20260925');BUILD.mkdir(exist_ok=True)
NAME='CS395_FinalReport_EfeHan_Keles_25September2026'
pdfmetrics.registerFont(TTFont('TNR','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
e92=json.loads((REPO/'evidence/e92_development.json').read_text())
def surface(name,w,h):
 path=BUILD/(name+'.pdf');can=canvas.Canvas(str(path),pagesize=(w,h));can.setFont('TNR',12);return can,path
def text(can,x,y,t,size=12):
 can.setFont('TNR',size)
 for i,line in enumerate(t.split('\n')):can.drawCentredString(x,y-i*size*1.18,line)
def box(can,x,y,w,h,t,size=12):
 can.rect(x,y,w,h);lines=t.count('\n')+1;text(can,x+w/2,y+h/2+(lines-1)*size*.59-size*.3,t,size)
def finish(can,path,name):
 can.save();convert_from_path(path,dpi=190)[0].save(ROOT/'figures'/(name+'.png'))
can,path=surface('false_alerts',560,235)
drawing=Drawing(560,230);bc=VerticalBarChart();bc.x=65;bc.y=40;bc.width=450;bc.height=155
bc.data=[[e92['reports'][cond][key]['binary_metrics']['confusion']['fp'] for cond in ['publisher_original','social_q75']] for key in ['old','E86_predecessor','new']]
bc.categoryAxis.categoryNames=['Original','Social-style JPEG75'];bc.categoryAxis.labels.fontName='TNR';bc.categoryAxis.labels.fontSize=12
bc.valueAxis.valueMin=0;bc.valueAxis.valueMax=80;bc.valueAxis.valueStep=20;bc.valueAxis.labels.fontName='TNR';bc.valueAxis.labels.fontSize=11
bc.barLabels.fontName='TNR';bc.barLabels.fontSize=12;bc.barLabelFormat='%d';bc.groupSpacing=25
for i,col in enumerate([colors.HexColor('#eeeeee'),colors.HexColor('#aaaaaa'),colors.HexColor('#444444')]):bc.bars[i].fillColor=col;bc.bars[i].strokeColor=colors.black
bc.barLabels.nudge=6;drawing.add(bc);renderPDF.draw(drawing,can,0,0)
text(can,280,218,'False alerts out of 160 real images',13)
for x0,col,t in [(85,'#eeeeee','E43 reference'),(255,'#aaaaaa','E86'),(370,'#444444','E92')]:
 can.setFillColor(colors.HexColor(col));can.rect(x0,5,14,9,fill=1);can.setFillColor(colors.black);can.setFont('TNR',11);can.drawString(x0+21,5,t)
finish(can,path,'false_alerts')
can,path=surface('organization',600,238)
box(can,225,187,150,35,'Board of Directors');box(can,446,187,130,35,'Internal Audit');can.line(375,204,446,204)
box(can,225,132,150,35,'General Manager');can.line(300,167,300,187)
labels=[('Commercial functions','Marketing and customer experience\nConsumer and corporate sales\nStrategy and wholesale'),('Technology functions','IT\nNetwork'),('Support and coordination','Finance, HR and procurement\nLegal, risk and communications\nInvestor relations and regions')]
for x0,(title,body) in zip([5,207,409],labels):
 box(can,x0,5,186,95,title+'\n\n'+body,10.4);can.line(x0+93,100,x0+93,117)
can.line(98,117,502,117);can.line(300,117,300,132);finish(can,path,'organization')
can,path=surface('pipeline',600,140)
for x0,t in [(3,'Validated\nphoto upload'),(155,'Original and\nfixed JPEG view'),(307,'Frozen encoders\nand E92 adaptation'),(459,'Evidence result\nand review warning')]:
 box(can,x0,54,135,56,t,12)
 if x0<450:
  can.line(x0+135,82,x0+150,82);can.line(x0+150,82,x0+145,86);can.line(x0+150,82,x0+145,78)
text(can,300,22,'Raw model scores remain separate from authenticity claims.',12);finish(can,path,'pipeline')
# Compact audited metric ledger for charts and later slide generation.
metrics={}
for cond,entry in e92['reports'].items():
 metrics[cond]={}
 for key in ['old','E86_predecessor','new']:
  b=entry[key]['binary_metrics'];metrics[cond][key]={k:b[k] for k in ['confusion','ai_recall','real_false_positive_rate','balanced_accuracy']}
(ROOT/'sources/verified_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
d=Document();s=d.sections[0];s.page_width=Inches(8.27);s.page_height=Inches(11.69)
s.top_margin=s.bottom_margin=s.left_margin=s.right_margin=Inches(1);s.header_distance=s.footer_distance=Inches(.45)
for style in d.styles:
 if style.type==1 or style.type==2:
  style.font.name='Times New Roman';style.font.size=Pt(12);style.font.color.rgb=RGBColor(0,0,0)
for st in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
 f=d.styles[st].paragraph_format;f.left_indent=f.right_indent=f.first_line_indent=Inches(0);f.line_spacing=2;f.space_before=f.space_after=Pt(0)
 f.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
for st in ['Title','Heading 1','Heading 2','Heading 3']:
 d.styles[st].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.LEFT;d.styles[st].font.bold=True
 d.styles[st].paragraph_format.keep_with_next=True
for st in ['Heading 1','Heading 2','Heading 3']:
 d.styles[st].paragraph_format.space_before=Pt(12)
 d.styles[st].paragraph_format.space_after=Pt(6)
# Eliminate theme color and font behavior from every named style.
for el in d.styles.element.xpath('.//w:pBdr'):
 el.getparent().remove(el)
for el in d.styles.element.xpath('.//w:color'):
 el.set(qn('w:val'),'000000')
 for a in ('themeColor','themeTint','themeShade'):el.attrib.pop(qn('w:'+a),None)
for el in d.styles.element.xpath('.//w:rFonts'):
 el.set(qn('w:ascii'),'Times New Roman');el.set(qn('w:hAnsi'),'Times New Roman');el.set(qn('w:eastAsia'),'Times New Roman');el.set(qn('w:cs'),'Times New Roman')
 for a in ('asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme'):el.attrib.pop(qn('w:'+a),None)
# Compact running head and page numbers, black throughout.
h=s.header.paragraphs[0];h.text='CS395 INTERNSHIP REPORT                                      PIXELPROOF';h.paragraph_format.line_spacing=1;h.alignment=WD_ALIGN_PARAGRAPH.LEFT
f=s.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.CENTER;f.paragraph_format.line_spacing=1
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');f._p.append(fld)
def paragraph(t,style=None):
 p=d.add_paragraph(t,style);p.paragraph_format.widow_control=True
 p.paragraph_format.line_spacing=2;p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
 return p
for t in ['Sabancı University','Faculty of Engineering and Natural Sciences','Computer Science and Engineering']:paragraph(t)
p=paragraph(c.TITLE,'Title');p.paragraph_format.space_before=Pt(54);p.paragraph_format.space_after=Pt(36)
paragraph('CS395 Internship Final Report')
paragraph('Efe Han Keleş');paragraph('Student ID: 31994')
for t in ['Company: Türk Telekomünikasyon A.Ş.','Supervisor: Önder Çelebi','Supervisor title: Director']:paragraph(t)
for t in ['Internship: 20 July–18 September 2026','Approved duration: 40 internship days','Format: Hybrid']:paragraph(t)
paragraph('Submission date: [TO COMPLETE]')
d.add_page_break();paragraph('Abstract','Title');paragraph(c.ABSTRACT)
# Two fixed TOC pages with a genuine Word TOC field and populated cache.
heads=[b for b in c.B if b['type']=='heading'];page_map={}
mp=ROOT/'sources/heading_pages.json'
if mp.exists():page_map=json.loads(mp.read_text())
d.add_page_break();paragraph('Table of Contents','Title')
start=d.add_paragraph();r=start.add_run();e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'begin');r._r.append(e)
r=start.add_run();e=OxmlElement('w:instrText');e.set(qn('xml:space'),'preserve');e.text=' TOC \\o "1-3" \\h \\z \\u ';r._r.append(e)
r=start.add_run();e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'separate');r._r.append(e)
start.paragraph_format.line_spacing=1;start.paragraph_format.space_after=Pt(0);start.paragraph_format.space_before=Pt(0)
for i,b in enumerate(heads):
 if i==23:d.add_page_break();paragraph('Table of Contents continued','Title')
 p=d.add_paragraph();p.paragraph_format.line_spacing=1.45;p.paragraph_format.tab_stops.add_tab_stop(Inches(6.27),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
 p.alignment=WD_ALIGN_PARAGRAPH.LEFT
 p.add_run(b['text']+'\t'+str(page_map.get(b['text'],'…')))
r=d.paragraphs[-1].add_run();e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'end');r._r.append(e)
for idx,b in enumerate(c.B):
 typ=b['type']
 if typ=='heading':
  p=paragraph(b['text'],f"Heading {b['level']}")
  if b.get('break'):p.paragraph_format.page_break_before=True
 elif typ=='paragraph':paragraph(b['text'])
 elif typ=='reference':
  p=paragraph(b['text']);p.paragraph_format.line_spacing=1;p.paragraph_format.space_after=Pt(12);p.paragraph_format.keep_together=True
  # Journal/series titles in italics as required by the office reference guidance.
  italics=['Advances in Neural Information Processing Systems','Proceedings of Machine Learning Research','Transactions on Machine Learning Research']
  for token in italics:
   if token in b['text']:
    a,z=b['text'].split(token,1);p.clear();p.add_run(a);p.add_run(token).italic=True;p.add_run(z);break
 elif typ=='figure':
  p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True;p.paragraph_format.line_spacing=1
  p.add_run().add_picture(str(ROOT/'figures'/b['file']),width=Inches(6.27))
  p=paragraph(f"Figure {b['number']}. {b['title']}",'Caption');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=False;p.paragraph_format.keep_together=True
 elif typ=='table':
  p=paragraph(f"Table {b['number']}. {b['title']}",'Caption');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
  table=d.add_table(rows=1,cols=len(b['headers']));table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
  widths=b.get('widths') or [6.27/len(b['headers'])]*len(b['headers'])
  for col,w in zip(table.columns,widths):col.width=Inches(w)
  for i,t in enumerate(b['headers']):table.rows[0].cells[i].text=t
  for row in b['rows']:
   for cell,t in zip(table.add_row().cells,row):cell.text=t
  rep=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(rep)
  for ri,row in enumerate(table.rows):
   cant=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(cant)
   for ci,cell in enumerate(row.cells):
    cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    pr=cell._tc.get_or_add_tcPr();bd=OxmlElement('w:tcBorders')
    for edge in ['top','left','bottom','right']:
     e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');bd.append(e)
    pr.append(bd)
    if ri==0:
     fill=OxmlElement('w:shd');fill.set(qn('w:fill'),'EFEFEF');pr.append(fill)
    for p in cell.paragraphs:
     p.paragraph_format.line_spacing=2;p.paragraph_format.keep_with_next=ri<len(table.rows)-1
     p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
     p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
     for r in p.runs:r.font.name='Times New Roman';r.font.size=Pt(12);r.font.color.rgb=RGBColor(0,0,0);r.bold=ri==0
  # Table layout deliberately keeps each compact table together.
for p in d.paragraphs:
 for r in p.runs:
  r.font.name='Times New Roman';r.font.size=Pt(12);r.font.color.rgb=RGBColor(0,0,0)
d.core_properties.author='Efe Han Keleş';d.core_properties.title=c.TITLE;d.core_properties.subject='CS395 internship report'
d.save(OUT/(NAME+'.docx'))
# Readable source and provenance inventory, without private inputs.
md=['# '+c.TITLE,'','## Abstract',c.ABSTRACT,'']
for b in c.B:
 if b['type']=='heading':md+=['#'*b['level']+' '+b['text'],'']
 elif b['type'] in ['paragraph','reference']:md+=[b['text'],'']
 elif b['type']=='figure':md +=[f"![Figure {b['number']}](figures/{b['file']})",f"Figure {b['number']}. {b['title']}",'']
 elif b['type']=='table':md +=[f"Table {b['number']}. {b['title']}",' | '.join(b['headers']),' | '.join(['---']*len(b['headers'])),*[' | '.join(row) for row in b['rows']],'']
(ROOT/'REPORT_EN.md').write_text('\n'.join(md))
print(json.dumps({'report':str(OUT/(NAME+'.docx')),'abstract_words':len(c.ABSTRACT.split()),'headings':len(heads),'body_words':sum(len(b.get('text','').split()) for b in c.B)},indent=2))
