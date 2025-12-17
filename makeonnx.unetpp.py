import torch
from models.UNetPlusPlus import UNetpp
from onnx_picker_base import OnnxSlidingWindowPicker


class Picker(OnnxSlidingWindowPicker):
    def __init__(self):
        super().__init__(UNetpp, ckpt_path="ckpt/china.unetpp.pt")


model = Picker()
model.eval()
input_names = ["wave"]
output_names = ["prob", "time"]
x = torch.randn([500000, 3])
torch.onnx.export(
    model,
    x,
    "pickers/unetpp.onnx",
    verbose=True,
    dynamic_axes={"wave": {0: "batch"}, "prob": {0: "batch"}, "time": {0: "batch"}},
    input_names=input_names,
    output_names=output_names,
    opset_version=11,
)
