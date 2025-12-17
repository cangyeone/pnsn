import torch
from models.UNet import UNet
from onnx_picker_base import OnnxSlidingWindowPicker


class Picker(OnnxSlidingWindowPicker):
    def __init__(self):
        super().__init__(UNet, ckpt_path="ckpt/china.unet.pt", seqlen=3072, overlap=1536)


model = Picker()
model.eval()
input_names = ["wave"]
output_names = ["prob", "time"]
x = torch.randn([500000, 3])
torch.onnx.export(
    model,
    x,
    "pickers/unet.onnx",
    verbose=True,
    dynamic_axes={"wave": {0: "batch"}, "prob": {0: "batch"}, "time": {0: "batch"}},
    input_names=input_names,
    output_names=output_names,
    opset_version=11,
)
