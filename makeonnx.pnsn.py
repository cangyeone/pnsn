import torch
from models.BRNNPNSN import BRNN
from onnx_picker_base import OnnxSlidingWindowPicker

class Picker(OnnxSlidingWindowPicker):
    def __init__(self):
        super().__init__(BRNN, ckpt_path="ckpt/china.rnn.pnsn.pt", norm="std", diff=False, seqlen=10240, overlap=1024)

model = Picker()
model.eval()
input_names = ["wave"]
output_names = ["prob", "time"]
x = torch.randn([500000, 3])
torch.onnx.export(
    model,
    x,
    "pickers/pnsn.v1.onnx",
    verbose=True,
    dynamic_axes={"wave": {0: "batch"}, "prob": {0: "batch"}, "time": {0: "batch"}},
    input_names=input_names,
    output_names=output_names,
    opset_version=17,
)
