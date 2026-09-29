// Run a copy from a private build directory linked to bundled node_modules.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
const ROOT=process.env.REPORT_ROOT;
const WORK=process.env.PRESENTATION_BUILD;
const SKILL=process.env.PRESENTATION_SKILL;
const PY=process.env.RUNTIME_PYTHON;
if (![ROOT,WORK,SKILL,PY].every(x=>x&&path.isAbsolute(x))) throw Error('Absolute runtime paths required');
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const M=JSON.parse(await fs.readFile(path.join(ROOT,'sources/verified_metrics.json'),'utf8'));
const FONT='Times New Roman';
const notes=[];
function txt(s,text,x=72,y=170,w=1136,h=95,size=30,bold=false,align='left'){
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=text;a.text.style={typeface:FONT,fontSize:size,bold,color:'#000000',autoFit:'none',alignment:align};return a;
}
function base(p,n,title,source='PixelProof research prototype'){
 const s=p.slides.add();s.background.fill='#FFFFFF';txt(s,`${n}. ${title}`,72,42,1136,98,42,true);
 txt(s,source,72,662,1060,32,18);txt(s,String(n),1150,662,60,30,18,false,'right');return s;
}
function speak(s,n,title,seconds,text,sources=''){s.speakerNotes.textFrame.setText(`Suggested time: ${seconds} seconds.\n\n${text}\n\nSources and optional detail (not spoken): ${sources}`);notes.push({n,title,seconds,text,sources});}
function table(s,caption,values,widths,y=245,height=270){
 txt(s,caption,72,y-65,1136,55,24,true,'center');
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:72,top:y,width:1136,height,columnWidths:widths,values});
 t.borders.assign({style:'solid',fill:'#999999',width:1});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
 const cell=t.getCell(r,c);cell.fill=r?'#FFFFFF':'#EEEEEE';cell.text.style={typeface:FONT,fontSize:27,color:'#000000',bold:r===0,alignment:'left',autoFit:'none'};
 }
 return t;
}
const p=Presentation.create({slideSize:{width:1280,height:720}});
let s=base(p,1,'PixelProof AI image detection');
txt(s,'Reducing false AI warnings on real photographs',72,198,1136,95,40,true);
txt(s,'Efe Han Keleş\nCS395 internship at Türk Telekom',72,335,1136,95,32);
txt(s,'20 July–18 September 2026\n40 approved internship days',72,452,1136,80,28);
txt(s,'Supervisor: Önder Çelebi, Director\nPlanning and Development Directorate',72,548,1136,68,25);
txt(s,'Submission date: 29 September 2026',72,622,1136,30,22);
speak(s,1,'Project purpose',35,'My internship project was PixelProof, a research prototype for AI image detection. I worked individually at Türk Telekom and asked my supervisor and mentors for advice. The main goal was to reduce false AI warnings on real photographs while keeping useful AI detection. I built a local demonstration and recorded both improvements and failed experiments. I will explain the method, the main result and the limits of that result.','Student-confirmed internship information. Report Sections 1, 3.1 and 4.2.');
s=base(p,2,'The problem and objective');
txt(s,'A real photograph can receive a false AI warning.',72,180,1136,90,36,true);
txt(s,'An AI image can also escape detection.',72,304,1136,80,36,true);
txt(s,'The objective was to reduce the first error\nwithout creating new misses.',72,452,1136,105,34);
speak(s,2,'Problem and objective',50,'The project has two important errors. A false positive means that a real photograph receives an AI warning. A false negative means that an AI image escapes detection. Lowering one error can increase the other, so I measured both. The harder question was whether the detector could work when the camera, generator or image processing changed. That is generalization. Model1 deals with the whole image. A smaller second module, Model2, explores where a local AI edit might be.','Report Sections 3.3 and 4.1.');
s=base(p,3,'Early accuracy did not transfer');
txt(s,'Table 1 shows the first major warning.',72,166,1136,55,32);
table(s,'Table 1. Historical ResNet-18 accuracy.',[['Image collection','Accuracy'],['Familiar test images','97.66%'],['Separate external collection','25.2%']],[790,346],300,210);
txt(s,'The gap led to checks of resolution, sources and labels.',72,565,1136,70,30,true);
speak(s,3,'What changed our approach',65,'At first, a pretrained ResNet model reached almost ninety-eight percent accuracy on familiar test images. But its accuracy fell to about twenty-five percent on a separate collection. These are historical results, and later auditing found problems in that external benchmark too. The lesson was still clear: one high score was not enough. I checked input resolution, found reversed labels in later datasets and repaired a procedure that used evaluation data to choose a threshold. These corrections changed some earlier conclusions. I kept the corrections in the records instead of reporting only the best numbers.','Keleş (2026), E1–E6, E10, E19b–E19c and E27. No independent final-performance claim from Table 1.');
s=base(p,4,'Transfer learning in PixelProof');
txt(s,'Existing models describe the image',72,166,1136,60,35,true);
txt(s,'Frozen DINOv2, CLIP and DEAR components',72,235,1136,62,30);
txt(s,'Our trained components use those descriptions',72,356,1136,60,35,true);
txt(s,'E92 adjusts the detection score',72,425,1136,62,30);
txt(s,'Python and PyTorch for modelling\nFastAPI and React for the local demo',72,548,1136,82,28);
speak(s,4,'Method and contribution',75,'I used transfer learning, which means reusing knowledge from existing models. The pretrained components convert the image into numerical descriptions called features. Their weights remain fixed. The project trains smaller components to use those features and adjust the detection score. This is more practical than training a large visual model from the beginning. The current demo model is called E92. My work covered choosing experiments, organizing data, reviewing results and building the application. I used AI coding assistance for implementation and documentation. After each experiment, the plan and logs recorded what worked, what failed and the next decision.','Radford et al. (2021), https://proceedings.mlr.press/v139/radford21a.html; Oquab et al. (2024), https://arxiv.org/abs/2304.07193; Kim et al. (2026), https://arxiv.org/abs/2606.10309v2; Keleş (2026), E51–E92. Published component benchmarks are not PixelProof guarantees.');
s=base(p,5,'Training and development data');
txt(s,'Table 2 separates learning from repeated comparison.',72,160,1136,65,31);
table(s,'Table 2. The main E92 data groups.',[['Purpose','Images','Use'],['Training','12,269','Learn the model'],['Development','160 real + 160 AI','Compare candidates']],[335,340,461],290,210);
txt(s,'Four training views per image: 49,076 views\nRepeated views do not create independent photographs.',72,548,1136,90,29);
speak(s,5,'Understanding the data',60,'E92 learned from twelve thousand two hundred sixty-nine training images. Each image had four processing versions, giving forty-nine thousand seventy-six training views. A compressed copy is still related to its original, so views are not independent photographs. The development collection was separate from training and contained one hundred sixty real images and one hundred sixty AI images. We used it repeatedly to compare candidates. That makes it useful for development, but limits its value as a final test. The downloaded archive size is also different from the number of images actually admitted to a particular experiment.','Keleş (2026), E92 training and E66 development records. Report Section 4.5.3. The real development images cover ten SIDD scenes; the AI groups are two previously seen families.');
s=base(p,6,'Fewer false warnings on real images');
txt(s,'Figure 1 compares the same 160 real images in each condition.',72,155,1136,55,28);
const chart=s.charts.add('bar',{position:{left:120,top:216,width:1040,height:330},categories:['Original','Resized + JPEG75'],series:[['old','Earlier model E43','#DDDDDD'],['new','Current model E92','#444444']].map(([key,name,fill])=>({name,fill,values:[M.publisher_original[key].confusion.fp,M.social_q75[key].confusion.fp]})),barOptions:{direction:'column',grouping:'clustered'},hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:22,fill:'#000000',typeface:FONT}},xAxis:{textStyle:{fontSize:23,fill:'#000000',typeface:FONT}},yAxis:{min:0,max:80,majorUnit:20,textStyle:{fontSize:20,fill:'#000000',typeface:FONT}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:25,fill:'#000000',typeface:FONT}}});
applyPresentationChartFont(chart,{fontFamily:FONT});
txt(s,'Figure 1. False AI warnings on reused development images.',72,558,1136,42,24,true,'center');
txt(s,'E92 detected 159/160 AI images in each condition.',72,610,1136,42,30,true);
speak(s,6,'Main result',80,'Figure 1 shows the main improvement. For real original images, false warnings fell from sixty-eight to zero out of one hundred sixty. After resizing and JPEG compression, they fell from sixty-eight to fourteen. The current model detected one hundred fifty-nine of one hundred sixty AI images in each condition. This is a useful improvement on these images. It does not mean that every new photograph will work. The comparisons use the same development images, and the real photographs come from only ten scenes. Also, these are results for each view separately. The website combines two views into one displayed outcome, which I will explain shortly.','Keleş (2026), evidence/e92_development.json. Transformation: long-side cap 1080 followed by JPEG quality 75. SIDD scene diversity: Abdelhamed et al. (2018), https://abdokamel.github.io/sidd/.');
s=base(p,7,'Full acceptance remained incomplete');
txt(s,'20/20 numerical development checks passed',72,176,1136,85,36,true);
txt(s,'One AI image detected by E43 became a new miss.',72,304,1136,85,32);
txt(s,'That separate protection rule failed.',72,420,1136,60,35,true);
txt(s,'A fresh, independent final test is still needed.',72,550,1136,70,32);
speak(s,7,'What the result proves',60,'Twenty out of twenty means ten numerical checks applied to two processing conditions. It does not mean twenty independent datasets or perfect classification. A separate rule required us to keep every AI image that the earlier model had detected. E92 newly missed one of those original AI images. Therefore it did not pass full acceptance, even though the overall numbers improved. We also reused the development set while making decisions. A stronger claim would need a final evaluation that had not influenced those decisions. This is why I describe E92 as a research prototype.','Keleş (2026), E92 acceptance report. Dwork et al. (2015), https://proceedings.neurips.cc/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html. The 20 criteria are project-defined and correlated.');
s=base(p,8,'The local web demonstration');
txt(s,'An upload receives one of the outcomes in Table 3.',72,163,1136,60,30);
table(s,'Table 3. How the demo communicates evidence.',[['Result','Meaning'],['AI evidence','The detector found a warning signal'],['No clear AI evidence','The detector found insufficient evidence'],['Uncertain','The checks do not support a clear outcome']],[370,766],288,245);
txt(s,'The score measures detector response.\nIt does not certify the photograph’s authenticity.',72,565,1136,80,30);
speak(s,8,'Demo and uncertainty',55,'The demo lets a user upload an image and read a simple result. It can report AI evidence, no clear AI evidence or uncertainty. The displayed score describes the detector response. It is not a verified probability that the image is AI. A low score therefore cannot prove that the photograph is authentic. Comparing the original with a processed copy helps reveal unstable responses. The software also validates uploads and checks model artifacts. These guards improve application behavior, while detection reliability still needs separate evaluation.','Report Section 4.5.6 and current-policy audit. Guo et al. (2017), https://proceedings.mlr.press/v70/guo17a.html. Optional Q&A: upper score cut ~7.94; lower ~1.15. Original ≥ upper retains AI alert, both E92 views < lower give no clear evidence, other cases uncertain. E43 is advisory. Paired development outcomes: real 139 clear/21 uncertain/0 AI; AI 159 alerts/1 uncertain.');
s=base(p,9,'Model2 remains experimental');
txt(s,'Task: locate an AI-edited region',72,173,1136,65,36,true);
txt(s,'Data: source overlap and unclear edit masks',72,275,1136,65,32);
txt(s,'Local generation also changed background pixels.\nControlled pilot: 16 reused images, one editor.',72,386,1136,115,32);
txt(s,'The candidate was rejected.',72,558,1136,65,34,true);
speak(s,9,'The second model',60,'Model2 asks where an AI edit is located. We found datasets, but source overlap and unclear masks prevented building a broad, validated training set. In our Stable Diffusion pilot, asking for a local edit also changed pixels outside the selected region. We therefore pasted only the generated region into the original and kept comparison controls. This gave us sixteen controlled examples from one editor, not broad coverage. Later training helped a new edit placement but harmed the original placement and increased false markings on real images. I rejected that candidate. These data and evaluation limits kept Model2 experimental.','Keleş (2026), E105/E109/E117, E122/E127 and E129–E135. Optional Q&A: 506/512 CocoGlide originals overlap protected ancestry; not proof of historical weight contamination. DiffSeg30k: 201 partial, 224 full-positive and 87 empty masks. E127: 11 accepted of 16 attempts, mean 99.9417% outside pixels changed and outside MAE 9.79/255; pixel change is not semantic change. Exact composites preserve outside pixels but introduce possible seam cues. E129 later passed all 16 engineering checks. E135 original/new-placement AUC .74→.70 / .57→.64; authentic false area 17.11%→25.74%. No ChatGPT pixel experiment or universal impossibility claim.');
s=base(p,10,'Outcome and next step');
txt(s,'Table 4 summarizes the outcome of the internship.',72,157,1136,65,31);
table(s,'Table 4. Delivered work and remaining evidence.',[['Delivered','Next step'],['Working local prototype','Independent final evaluation'],['Fewer development false warnings','More cameras and generators'],['Recorded experiments and corrections','Broader Model2 controls']],[568,568],282,245);
txt(s,'My main lesson: check what a good score actually proves.',72,575,1136,65,32,true);
speak(s,10,'Conclusion',60,'The internship produced a working local prototype, trained adaptation components and a record of experiments and corrections. The strongest measured gain was fewer false warnings on the defined development collection. Broader reliability remains unresolved. For example, the owner gallery still produced false warnings and had already influenced development. The next step is evaluation on independent sources with fixed decision rules. Model2 needs broader controls too. Personally, I learned how to manage an individual project, narrow its scope and explain what the evidence supports. Building a working demo was useful, but understanding its limits was just as important. Thank you.','Keleş (2026), E95, E139, E146–E152. Optional Q&A: E92 gallery false alerts 9/206 originals, 18/206 processed. Earlier E43 comprehensive E49 test failed 11/20 on 2,000 parents; it is not E92 final performance. No production rollout or measured company impact.');
s=base(p,11,'References for the method','Reference appendix');
const refs1=[['Radford et al. (2021)','Learning transferable visual models from natural language supervision. PMLR 139.'],['Oquab et al. (2024)','DINOv2: Learning robust visual features without supervision. TMLR.'],['Kim et al. (2026)','Dissect and prune: Enhancing robustness in AI-generated image detection. ICML.'],['Ojha et al. (2023)','Towards universal fake image detectors that generalize across generative models. CVPR.']];
refs1.forEach(([a,b],i)=>{txt(s,a,72,175+i*112,1136,38,29,true);txt(s,b,72,218+i*112,1136,64,26);});
speak(s,11,'Method references',0,'Reference appendix. The report provides the full bibliography.','https://proceedings.mlr.press/v139/radford21a.html\nhttps://arxiv.org/abs/2304.07193\nhttps://arxiv.org/abs/2606.10309v2\nhttps://arxiv.org/abs/2302.10174');
s=base(p,12,'References for data and evaluation','Reference appendix');
const refs2=[['Abdelhamed et al. (2018)','A high-quality denoising dataset for smartphone cameras. CVPR.'],['Guo et al. (2017)','On calibration of modern neural networks. PMLR 70.'],['Dwork et al. (2015)','Generalization in adaptive data analysis and holdout reuse. NeurIPS 28.'],['Keleş (2026)','PixelProof experiment, history and dataset records. GitHub, commit 497d45c.']];
refs2.forEach(([a,b],i)=>{txt(s,a,72,175+i*112,1136,38,29,true);txt(s,b,72,218+i*112,1136,64,26);});
speak(s,12,'Data and evaluation references',0,'Reference appendix. The report gives fourteen full entries. These slides list the sources most relevant to the short talk.','https://abdokamel.github.io/sidd/\nhttps://proceedings.mlr.press/v70/guo17a.html\nhttps://proceedings.neurips.cc/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html\nhttps://github.com/EfeHanKeles346/ai-image-detector/tree/497d45c');
const d=Presentation.create({slideSize:{width:1280,height:720}});
s=base(d,1,'PixelProof AI image detection','');
txt(s,'Türk Telekom\nEfe Han Keleş, Computer Science and Engineering',72,143,1136,82,29,true);
txt(s,'20 July–18 September 2026, 40 approved internship days\nFatih Sultan Mehmet Mah., Balkan Cad. No:49, 34771 Ümraniye / İstanbul',72,235,1136,77,25);
txt(s,'Objective',72,336,250,48,31,true);
txt(s,'Reduce false AI warnings on real photographs\nwhile preserving AI detection. Explore local AI edits.',345,336,865,86,29);
txt(s,'Outcome',72,438,250,48,31,true);
txt(s,'Working local demo and documented experiments.\nOn reused development data: 159/160 AI detected per condition.\nReal false warnings: 0/160 original, 14/160 after processing.',345,438,865,118,27);
txt(s,'Limit',72,570,250,45,31,true);
txt(s,'One new AI miss prevented full acceptance.\nBroader reliability and Model2 remain unproven.',345,570,865,72,27);
s.speakerNotes.textFrame.setText('Submission date: 29 September 2026. Supervisor: Önder Çelebi, Director of the Planning and Development Directorate. E92 used 12,269 training parents, four views each (49,076 views). The separate but reused development set contains 160 real and 160 AI parents. Processed condition means long-side cap 1080 then JPEG75. Twenty numerical development checks passed, but E92 newly missed one original AI image detected by E43. No independent final-test success or production deployment is claimed. Model2 remains experimental. Sources: Keleş (2026), https://github.com/EfeHanKeles346/ai-image-detector/tree/497d45c, E92 evidence and E135 localization result. Student-confirmed internship information. Full sources in report Section 8.');
txt(s,'Supervisor: Önder Çelebi, Director   Submission date: 29 September 2026',72,670,1136,26,19);
await fs.mkdir(WORK,{recursive:true});
if(process.env.PRESENTATION_KIND!=='Digest') await fs.writeFile(path.join(ROOT,'sources/presentation_notes.json'),JSON.stringify(notes,null,2)+'\n');
for(const [kind,pres,count,tables,charts] of [['Presentation',p,12,[3,5,8,10],[6]],['Digest',d,1,[],[]]]){
 if(process.env.PRESENTATION_KIND && process.env.PRESENTATION_KIND!==kind) continue;
 const wd=path.join(WORK,kind);await fs.mkdir(path.join(wd,'.codex-finalizer'),{recursive:true});
 const candidate=path.join(wd,'.codex-finalizer/candidate.pptx');await(await PresentationFile.exportPptx(pres)).save(candidate);
 await fs.mkdir(path.join(wd,'output'),{recursive:true});
 const final=path.join(wd,'output',`CS395_${kind}_EfeHan_Keles_29September2026.pptx`);
 const result=await finalizePresentation({workspaceDir:wd,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:charts,explicitTotalSlideCount:count,fontPolicy:{basis:'user_request',families:[FONT]},materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,receiptPath:path.join(wd,'.codex-finalizer/validation.json')});
 console.log(kind,JSON.stringify({finalPath:result.finalPath,sha256:result.finalSha256,warnings:result.presentationLayout?.warnings}));
 await fs.copyFile(final,path.join(ROOT,'deliverables',path.basename(final)));
}
console.log('Speaking time seconds',notes.reduce((a,n)=>a+n.seconds,0));
