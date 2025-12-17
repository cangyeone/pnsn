import torch
from models.EQTransformer import EQTransformer
from onnx_picker_base import OnnxSlidingWindowPicker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained = EQTransformer.from_pretrained("stead").to(device)
pretrained_state = {k: v.cpu() for k, v in pretrained.state_dict().items()}


class Picker(OnnxSlidingWindowPicker):
    def __init__(self):
        super().__init__(EQTransformer, state_dict=pretrained_state, seqlen=6000, overlap=3000)


model = Picker()
model.eval()
input_names = ["wave"]
output_names = ["prob", "time"]
x = torch.randn([500000, 3])
torch.onnx.export(
    model,
    x,
    "pickers/eqtransformer.stead.onnx",
    verbose=True,
    dynamic_axes={"wave": {0: "batch"}, "prob": {0: "batch"}, "time": {0: "batch"}},
    input_names=input_names,
    output_names=output_names,
    opset_version=11,
)
