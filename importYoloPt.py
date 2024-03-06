from models.experimental import attempt_load
from utils.torch_utils import select_device

WEIGHTS = "D:\Smoke\SmokeDetectionMasterPython\smoke.pt"
def get_model():
    device = select_device('')
    half = device.type != 'cpu'
    model = attempt_load(WEIGHTS, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()
    return model, device, half, stride, names
