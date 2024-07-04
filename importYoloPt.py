import torch

from models.experimental import attempt_load
from utils.torch_utils import select_device


def get_model(WEIGHTS: str):
    device = select_device('')
    half = device.type != 'cpu'
    model = attempt_load(WEIGHTS, map_location=device)
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()
    model.eval()
    with torch.no_grad():
        warmup_input = torch.rand(1, 3, stride * 32, stride * 32).to(device).type_as(next(model.parameters()))
        model(warmup_input.half() if half else warmup_input)
    return model, device, half, stride, names
