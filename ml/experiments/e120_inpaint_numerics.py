"""Locate nonfinite inpainting tensors before rendering/safety checks; fixed precision ablation."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image
from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import resource_check
from experiments import e119_inpainting_pilot as previous, e118_inpaint_assets as assets
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e120'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'
BRANCHES = ('fp16_sliced', 'fp16_sdpa', 'fp32_sliced')


class NonfiniteTensor(ValueError):
    pass


def check_array(value, stage):
    # CPU version used by output/receipt checks and focused tests.
    if not np.isfinite(np.asarray(value)).all(): raise NonfiniteTensor(stage)


def freeze():
    previous.validate(); assets.validate()
    if digest(previous.ROOT/'result.json') != read(EVIDENCE/'e119_inpainting_pilot.json')['result_sha256'] or \
            read(previous.ROOT/'result.json')['passed']:
        raise ValueError('Complete failed E119 required')
    files = [Path(__file__), previous.CONTRACT, previous.ROOT/'prepared.json', previous.ROOT/'result.json',
        EVIDENCE/'e119_runtime_dependencies.json', ML_ROOT/'requirements-model2-pilot.lock',
        DATA_ROOT/'e119_pipeline/e119_inpainting_pilot_generate.log', assets.ROOT/'download.json']
    c = {'state': 'E120_fixed_numerical_probe_registered',
        'inputs': read(previous.CONTRACT)['inputs'] | {str(p): digest(p) for p in files},
        'parents': [0,1], 'branches': list(BRANCHES),
        'selection': 'Original first successful and first failed E119 parents, unchanged original/mask/prompt/seeds119000/119001. Diagnostic only; no replacement or image-quality optimization.',
        'changes': 'Three fixed branches: reproduce fp16+sliced; change only attention to SDPA at fp16; change only full-pipeline compute precision to float32 with sliced attention. Same fp16 source weights cast to float32, not new weights. Fresh pipeline per branch, parent0 then1, safety checker always retained.',
        'instrumentation': 'Check finite text encoder outputs, VAE posterior parameters/decode, UNet outputs, every denoising latent and decoded image before safety evaluation. Stop each invalid case at its first nonfinite boundary; never treat NaN-generated black images as semantic safety findings.',
        'choice_rule': 'If both fixed fp16_sdpa cases pass, use that implementation for a separately registered complete16-parent replay; otherwise require both fp32_sliced cases to pass. If neither passes, stop and diagnose. Two cases are not complete-pilot acceptance.',
        'max_seconds': 1800, 'mps_limit_bytes': 10*1024**3, 'downloads': 0, 'detector_scores': 0,
        'limits': 'An upstream SDXL+offload+slicing issue is a mechanism lead, not proof for this SD1.5/no-offload run. No safety-filter bypass, detector training, test access or serving change.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT,c)
    write_once(EVIDENCE/'e120_numerics_contract.json',{k:v for k,v in c.items() if k!='inputs'}|{'contract_sha256':digest(CONTRACT)})
    return {'branches': list(BRANCHES), 'parents': [0,1]}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e120_numerics_contract.json')['contract_sha256']:raise ValueError('Probe contract differs')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('Frozen numerical-probe input differs')
    previous.validate(); return c


def instrument(pipe, torch):
    counts = {}
    def finite(value, stage):
        if isinstance(value, torch.Tensor):
            counts[stage] = counts.get(stage,0)+1
            if not bool(torch.isfinite(value).all().item()): raise NonfiniteTensor(stage)
        elif isinstance(value, dict):
            for v in value.values(): finite(v,stage)
        elif isinstance(value,(tuple,list)):
            for v in value: finite(v,stage)
    for name in ('text_encoder','unet'):
        getattr(pipe,name).register_forward_hook(lambda m,a,out,stage=name: finite(out,stage))
    encode = pipe.vae.encode; decode = pipe.vae.decode; safety = pipe.run_safety_checker
    def checked_encode(*args,**kwargs):
        out=encode(*args,**kwargs); posterior=out.latent_dist if hasattr(out,'latent_dist') else out[0]
        finite(posterior.parameters,'vae_encode'); return out
    def checked_decode(*args,**kwargs):
        out=decode(*args,**kwargs); finite(out,'vae_decode'); return out
    def checked_safety(image,*args,**kwargs):
        finite(image,'before_safety'); return safety(image,*args,**kwargs)
    pipe.vae.encode=checked_encode; pipe.vae.decode=checked_decode; pipe.run_safety_checker=checked_safety
    return finite,counts


def run_cases(branch, rows, destination, deadline, memory_limit):
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    if branch not in BRANCHES: raise ValueError('Unregistered precision branch')
    torch.set_num_threads(1); resource_check(deadline)
    pipe=StableDiffusionInpaintPipeline.from_pretrained(str(assets.ROOT/'model'), local_files_only=True,
        dtype=torch.float32 if branch=='fp32_sliced' else torch.float16, variant='fp16',
        use_safetensors=True, low_cpu_mem_usage=False).to('mps')
    if pipe.safety_checker is None:raise ValueError('Safety checker missing')
    if branch!='fp16_sdpa':pipe.enable_attention_slicing()
    else:
        from diffusers.models.attention_processor import AttnProcessor2_0
        pipe.unet.set_attn_processor(AttnProcessor2_0())
        pipe.vae.set_attn_processor(AttnProcessor2_0())
    pipe.set_progress_bar_config(disable=True); finite,counts=instrument(pipe,torch); peak=0; results=[]
    def step(pipeline,index,timestep,kwargs):
        nonlocal peak
        resource_check(deadline); peak=max(peak,torch.mps.driver_allocated_memory())
        if peak>memory_limit:raise RuntimeError('Numerical probe MPS limit')
        finite(kwargs['latents'],'denoising_latents');return kwargs
    for row in rows:
        resource_check(deadline); start=time.monotonic(); counts.clear()
        for f in row['files'].values():
            if digest(f['path'])!=f['sha256']:raise ValueError('Pilot input pixels differ')
        with Image.open(row['files']['original']['path']) as im:original=im.convert('RGB')
        with Image.open(row['files']['mask']['path']) as im:mask=im.convert('L')
        dest=destination/branch/f'{row["index"]:03d}'; dest.mkdir(parents=True,exist_ok=True)
        if (dest/'case.json').exists():raise FileExistsError('Case already executed; no hidden reroll')
        result={'index':row['index'],'branch':branch,'seed':row['seed'],'passed':False}
        try:
            with torch.inference_mode():
                output=pipe(prompt=read(previous.CONTRACT)['generation']['prompt'],image=original,mask_image=mask,
                    height=512,width=512,num_inference_steps=30,guidance_scale=7.5,strength=1.0,num_images_per_prompt=1,
                    generator=torch.Generator(device='cpu').manual_seed(row['seed']),callback_on_step_end=step)
            generated=output.images[0].convert('RGB'); check_array(generated,'rendered_image')
            edited=previous.composite(original,generated,mask); generated.save(dest/'raw.png');edited.save(dest/'composite.png')
            a=np.asarray(original);b=np.asarray(generated);m=np.asarray(mask)>0;diff=np.any(a!=b,axis=2)
            flags=output.nsfw_content_detected
            checks={'safety_check_negative':flags is not None and len(flags)==1 and not bool(flags[0]),
                'nonblank_output':bool(b.std()>1),'masked_pixels_changed':bool(diff[m].mean()>=.95),
                'background_exact':bool(np.array_equal(np.asarray(edited)[~m],a[~m]))}
            result.update(checks=checks,passed=all(checks.values()),raw_sha256=digest(dest/'raw.png'),
                composite_sha256=digest(dest/'composite.png'),raw_background_changed_fraction=float(diff[~m].mean()))
        except NonfiniteTensor as exc:
            result.update(first_nonfinite_stage=str(exc),rendered_images=0,safety_classification_performed=False)
        result.update(seconds=time.monotonic()-start,finite_boundary_counts=dict(counts))
        write_once(dest/'case.json',result);results.append(result)
        print(json.dumps({'numerical_case':branch,'index':row['index'],'passed':result['passed'],
            'first_nonfinite_stage':result.get('first_nonfinite_stage')}),flush=True)
    del pipe;gc.collect();torch.mps.empty_cache()
    return results,peak


def probe():
    c=validate()
    if (ROOT/'report.json').exists():raise FileExistsError('Numerical probe complete')
    for r in read(assets.CONTRACT)['files']:assets.verify(assets.ROOT/'model'/r['rfilename'],r)
    rows=read(previous.ROOT/'prepared.json')['rows'][:2];start=time.monotonic();results={};peaks={}
    for branch in BRANCHES:results[branch],peaks[branch]=run_cases(branch,rows,ROOT,start+c['max_seconds'],c['mps_limit_bytes'])
    passes={b:all(r['passed'] for r in rs) for b,rs in results.items()}
    chosen=next((b for b in ('fp16_sdpa','fp32_sliced') if passes[b]),None)
    result={'state':'E120_fixed_numerical_probe_complete','contract_sha256':digest(CONTRACT),'results':results,
        'branch_passes':passes,'next_complete_replay_branch':chosen,'peak_mps_by_branch':peaks,
        'seconds':time.monotonic()-start,'detector_scores':0,'training_allowed':False,'limits':c['limits']}
    write_once(ROOT/'report.json',result);write_once(EVIDENCE/'e120_inpaint_numerics.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','probe']);args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def denied(*a,**kw):raise RuntimeError('Numerical probe is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'probe':probe}[args.stage](),indent=2))
