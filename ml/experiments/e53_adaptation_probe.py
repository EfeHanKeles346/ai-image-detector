"""Bounded local-backbone differentiation/resource probe, not a fitted candidate."""
import hashlib
import json
import os
import time

import joblib
import numpy as np

from experiments import e42_features as dino
from experiments.e51_pipeline import prepare
from experiments.e53_offline import ROOT, EVIDENCE, contract, digest, fixed_write
from pixelproof.project_paths import DATA_ROOT


def differentiable_aggregate(tokens):
    """Same crop population mean/std layout as E42, retaining autograd."""
    import torch
    if tokens.ndim != 3 or len(tokens) % 3 or tokens.shape[1] != 4:
        raise ValueError('expected three crops and four blocks per view')
    grouped = tokens.reshape(-1, 3, 4, tokens.shape[-1])
    return torch.cat((grouped.mean(dim=1).flatten(1),
                      grouped.std(dim=1, correction=0).flatten(1)), dim=1)


def probe():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    value, binding = contract()
    if (EVIDENCE/'e53_adaptation_probe.json').exists():
        raise FileExistsError('resource probe already complete')
    selected = []
    for label in (0, 1):
        selected.extend(sorted((r for r in value['rows'] if r['label'] == label),
            key=lambda r: hashlib.sha256(f"E53_RESOURCE|{r['parent_id']}".encode()).hexdigest())[:4])
    protocol = {'base_contract_sha256': binding, 'code_sha256': digest(__file__),
                'parents': [r['parent_id'] for r in selected], 'batch_parents': 8,
                'trainable': 'last two DINO blocks plus copied binary head',
                'optimizer': 'AdamW', 'learning_rate': 1e-5, 'weight_decay': .01,
                'warmup_steps': 1, 'timed_steps': 3, 'saved_candidate': False,
                'purpose': 'resource/gradient feasibility only, not a quality experiment'}
    fixed_write(ROOT/'adaptation_probe_contract.json', protocol)
    headpath = DATA_ROOT/'e51/training/e51_A.joblib'
    expected = json.loads((EVIDENCE/'e51_cal_result.json').read_text())['candidates']['A']['artifact_sha256']
    if digest(headpath) != expected:
        raise ValueError('reference head changed')
    sklearn_head = joblib.load(headpath)['head']
    scaler, classifier = sklearn_head.steps[0][1], sklearn_head.steps[1][1]
    model, means, stds, _ = dino._load_small()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.blocks[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    head = torch.nn.Linear(3072, 1).to(device)
    coefficient = classifier.coef_ / scaler.scale_
    intercept = classifier.intercept_ - coefficient @ scaler.mean_
    with torch.no_grad():
        head.weight.copy_(torch.tensor(coefficient, device=device, dtype=torch.float32))
        head.bias.copy_(torch.tensor(intercept, device=device, dtype=torch.float32))
    arrays = np.stack([crop for row in selected for crop in prepare(
        dict(row, role='TRAIN', condition='clean'))[0]])
    inputs = torch.from_numpy(arrays).permute(0, 3, 1, 2).to(device, dtype=torch.float32)/255.
    inputs = (inputs-torch.tensor(means, device=device).view(1, 3, 1, 1))/torch.tensor(stds, device=device).view(1, 3, 1, 1)
    labels = torch.tensor([r['label'] for r in selected], device=device, dtype=torch.float32)

    def features():
        blocks = model.forward_intermediates(inputs, indices=list(dino.BLOCKS['small']),
            return_prefix_tokens=True, norm=True, intermediates_only=True)
        tokens = torch.stack([item[1][:, 0, :] for item in blocks], dim=1)
        return tokens, differentiable_aggregate(tokens)

    def frozen_digest():
        h = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                h.update(name.encode()); h.update(parameter.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def sync():
        if device.type == 'mps':
            torch.mps.synchronize()

    before = frozen_digest()
    with torch.no_grad():
        tokens, aggregated = features()
        reference = dino.aggregate_tokens(tokens.cpu().numpy(), len(selected))
        feature_error = float(np.max(np.abs(reference-aggregated.cpu().numpy())))
        score_error = float(np.max(np.abs(sklearn_head.predict_proba(reference)[:, 1]
                                         -torch.sigmoid(head(aggregated).flatten()).cpu().numpy())))
    if max(feature_error, score_error) > 5e-5:
        raise ValueError('differentiable implementation failed parity before optimization')
    trainable = [p for p in (*model.parameters(), *head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=.01)
    durations, memory = [], []
    for step in range(4):
        sync(); started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        _, aggregated = features()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(aggregated).flatten(), labels)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite resource-probe loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
            raise ValueError('missing/nonfinite trainable gradient')
        optimizer.step(); sync()
        if step:
            durations.append(time.perf_counter()-started)
        if device.type == 'mps':
            memory.append({'current_bytes': torch.mps.current_allocated_memory(),
                           'driver_bytes': torch.mps.driver_allocated_memory()})
    after = frozen_digest()
    if before != after:
        raise ValueError('frozen parameters changed')
    result = {'state': 'resource_probe_complete', 'contract_sha256': digest(ROOT/'adaptation_probe_contract.json'),
              'device': str(device), 'parents': len(selected), 'crops_per_step': len(arrays),
              'trainable_parameters': sum(p.numel() for p in trainable),
              'total_backbone_parameters': sum(p.numel() for p in model.parameters()),
              'pre_step_feature_error': feature_error, 'pre_step_score_error': score_error,
              'finite_gradients_all_steps': True, 'frozen_parameter_sha256': before,
              'frozen_parameters_unchanged': True, 'timed_step_seconds': durations,
              'median_step_seconds': float(np.median(durations)), 'sampled_mps_memory': memory,
              'candidate_saved': False, 'serving_changed': False, 'quality_improvement_claimed': False,
              'limits': ['Eight-parent repeated batch only, not full-data training or validation.',
                         'Memory sampled after steps is not true allocator peak.',
                         'Feature extraction/decoding and evaluation cost are excluded from step timing.']}
    fixed_write(EVIDENCE/'e53_adaptation_probe.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(probe(), indent=2))
