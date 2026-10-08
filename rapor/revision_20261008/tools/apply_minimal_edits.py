from pathlib import Path
from copy import deepcopy
from docx import Document
import json,sys,hashlib
src=Path(sys.argv[1]);out=Path(sys.argv[2]);d=Document(src);changes=[]
repls=[('Radford et al., 2021).','(Radford et al., 2021).'),('in every experiment','in some experiments'),('but previous model missed','but current model missed')]
for before,after in repls:
 matches=[p for p in d.paragraphs if before in p.text]
 assert len(matches)==1,(before,len(matches))
 p=matches[0];old=p.text;new=old.replace(before,after)
 # Character-local replacement across runs preserves untouched formatting.
 pos=old.index(before);end=pos+len(before);cursor=0;inserted=False
 for run in p.runs:
  t=run.text;a=cursor;b=a+len(t);cursor=b
  if b<=pos or a>=end:continue
  left=t[:max(0,pos-a)];right=t[max(0,end-a):] if b>end else ''
  run.text=left+(after if not inserted else '')+right;inserted=True
 assert p.text==new
 changes.append({'before':old,'after':new})
duplicate='Guo et al. (2017) explain that a high model score does not always mean a high chance of being correct. Calibration helps confidence scores match actual accuracy better. In our web demo, the score is not shown as the probability of an image being AI generated. Choosing a threshold does not turn the score into a probability.'
ps=[p for p in d.paragraphs if p.text==duplicate];assert len(ps)==1
p=ps[0];p._p.getparent().remove(p._p);changes.append({'before':duplicate,'after':None})
out.parent.mkdir(parents=True,exist_ok=True);# Keep inherited black typography, including newly typed author text.
from docx.shared import RGBColor
for p in d.paragraphs:
 for run in p.runs:run.font.color.rgb=RGBColor(0,0,0)
d.save(out)
(out.parent/'minimal_changes.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'changes':changes},ensure_ascii=False,indent=2)+'\n')
print(out)
