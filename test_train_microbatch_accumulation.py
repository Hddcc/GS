import copy
import importlib.machinery
import sys
import types

import torch
import torch.nn as nn

try:
    import tqdm  # noqa: F401
except ImportError:
    tqdm_module = types.ModuleType('tqdm')
    tqdm_module.__spec__ = importlib.machinery.ModuleSpec('tqdm', None)
    tqdm_module.tqdm = lambda iterable, **kwargs: iterable
    sys.modules['tqdm'] = tqdm_module

for module_name in ('datasets', 'models'):
    module = types.ModuleType(module_name)
    module.__spec__ = importlib.machinery.ModuleSpec(module_name, None)
    sys.modules[module_name] = module
test_module = types.ModuleType('test')
test_module.__spec__ = importlib.machinery.ModuleSpec('test', None)
test_module.eval_psnr = None
sys.modules['test'] = test_module

import train_gaussian


class ToyImplicitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)

    def forward(self, inp, coord, scale, cell):
        del scale, cell
        feature = inp.mean(dim=(-2, -1))
        rgb = self.projection(feature)
        return rgb[:, None, :].expand(-1, coord.shape[1], -1)


def make_batch():
    generator = torch.Generator().manual_seed(17)
    return {
        'inp': torch.rand(12, 3, 4, 4, generator=generator),
        'coord': torch.rand(12, 7, 2, generator=generator),
        'scale': torch.tensor([1.5] * 4 + [4.0] * 4 + [8.0] * 4),
        'cell': torch.rand(12, 7, 2, generator=generator),
        'gt': torch.rand(12, 7, 3, generator=generator),
    }


def train_once(initial_state, batch, microbatch_size):
    model = ToyImplicitModel()
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    result = train_gaussian.train(
        [copy.deepcopy(batch)], model, optimizer,
        microbatch_size=microbatch_size,
    )
    return model.state_dict(), result['loss']


def main():
    ranges = train_gaussian.microbatch_ranges(12, 4)
    if ranges != ((0, 4), (4, 8), (8, 12)):
        raise AssertionError('Unexpected microbatch partition.')
    batch = make_batch()
    for start, end in ranges:
        if torch.unique(batch['scale'][start:end]).numel() != 1:
            raise AssertionError('A microbatch contains mixed scales.')

    torch.manual_seed(23)
    initial_state = ToyImplicitModel().state_dict()
    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    try:
        full_state, full_loss = train_once(initial_state, batch, None)
        micro_state, micro_loss = train_once(initial_state, batch, 4)
    finally:
        torch.Tensor.cuda = original_cuda

    loss_error = abs(full_loss - micro_loss)
    parameter_error = max(
        (full_state[name] - micro_state[name]).abs().max().item()
        for name in full_state
    )
    print('full loss: {:.10f}'.format(full_loss))
    print('microbatch loss: {:.10f}'.format(micro_loss))
    print('loss error: {:.3e}'.format(loss_error))
    print('parameter update error: {:.3e}'.format(parameter_error))
    if loss_error > 1e-7 or parameter_error > 1e-7:
        raise AssertionError('Microbatch accumulation changed the update.')
    print('MICROBATCH ACCUMULATION TEST PASSED')


if __name__ == '__main__':
    train_gaussian.config = {
        'data_norm': {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]},
        }
    }
    main()
