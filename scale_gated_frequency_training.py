import hashlib
from pathlib import Path

import torch


ADAPTATION_PREFIXES = ('frequency_encoder.', 'residual_head.', 'scale_gate.')


def initialize_from_baseline(model, path, seed):
    checkpoint = torch.load(path, map_location='cpu')
    spec = checkpoint['model']
    if spec['name'] != 'gaussian-splatter':
        raise ValueError('Warm start requires a gaussian-splatter baseline checkpoint.')
    if checkpoint.get('seed') != seed:
        raise ValueError('Baseline checkpoint seed does not match this run.')
    source = spec['sd']
    target = model.state_dict()
    shared = {key for key in target if not key.startswith(
        ADAPTATION_PREFIXES + ('residual_gate_logit',)
    )}
    if set(source) != shared:
        raise ValueError('Baseline keys do not exactly match the shared backbone.')
    for key in shared:
        if source[key].shape != target[key].shape:
            raise ValueError('Baseline shape mismatch: ' + key)
    target.update(source)
    model.load_state_dict(target, strict=True)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {'path': str(Path(path).resolve()), 'sha256': digest,
            'source_epoch': checkpoint['epoch'], 'source_seed': seed}


def optimizer_groups(model, backbone_lr, residual_lr):
    adaptation = list(model.adaptation_parameters()) if hasattr(
        model, 'adaptation_parameters'
    ) else []
    adaptation_ids = {id(parameter) for parameter in adaptation}
    backbone = [parameter for parameter in model.parameters()
                if id(parameter) not in adaptation_ids and parameter.requires_grad]
    groups = [{'params': backbone, 'lr': backbone_lr, 'name': 'backbone'}]
    if adaptation:
        groups.append({'params': adaptation, 'lr': residual_lr, 'name': 'residual'})
    return groups


def set_training_stage(model, epoch, freeze_epochs):
    model = model.module if isinstance(model, torch.nn.DataParallel) else model
    has_adaptation = hasattr(model, 'adaptation_parameters')
    for name, parameter in model.named_parameters():
        if name == 'residual_gate_logit':
            parameter.requires_grad_(False)
        elif not name.startswith(ADAPTATION_PREFIXES):
            parameter.requires_grad_(not has_adaptation or epoch > freeze_epochs)
    return 'residual-only' if has_adaptation and epoch <= freeze_epochs else 'joint'
