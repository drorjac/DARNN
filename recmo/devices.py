"""Explicit execution devices and synchronized accelerator timing."""
import platform

import torch


def resolve_device(requested='cpu'):
    """Auto prefers CUDA, then MPS, then CPU; explicit requests never fall back."""
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError('Choose cpu, auto, mps, cuda, or cuda:N.') from error
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable. Use a CUDA-enabled PyTorch build on a supported NVIDIA host, or --device cpu.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {device.index} does not exist.')
    elif device.type == 'mps':
        if device.index is not None or not torch.backends.mps.is_available():
            raise ValueError('MPS unavailable. Check the installed PyTorch/macOS compatibility, or use --device cpu.')
    elif device.type != 'cpu' or device.index is not None:
        raise ValueError('Choose cpu, auto, mps, cuda, or cuda:N.')
    return device


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def device_info(device):
    name = platform.processor() or platform.machine()
    if device.type == 'cuda':
        name = torch.cuda.get_device_name(device)
    elif device.type == 'mps':
        name = 'Apple Metal GPU'
    return {'device': str(device), 'device_name': name, 'host': platform.node(),
            'platform': platform.platform(), 'torch_version': str(torch.__version__)}


def cpu_state(module):
    """Store portable model parameters independently of the training device."""
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}
