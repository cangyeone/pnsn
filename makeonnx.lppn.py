import torch
from onnx_picker_base import OnnxSlidingWindowPicker

MODEL_NAME = "lppnm"

if MODEL_NAME == "lppnt":
    from models.LPPNT import Model
    CKPT_PATH = "ckpt/china.lppnt.pt"
elif MODEL_NAME == "lppnl":
    from models.LPPNL import Model
    CKPT_PATH = "ckpt/china.lppnl.pt"
else:
    from models.LPPNM import Model
    CKPT_PATH = "ckpt/china.lppnm.pt"


class Picker(OnnxSlidingWindowPicker):
    def __init__(self):
        super().__init__(Model, ckpt_path=CKPT_PATH)
        self.n_stride = 8

    def forward(self, x):
        device = x.device
        with torch.no_grad():
            wave, batchlen = self.window_and_normalize(x, device)
            wave = wave.unsqueeze(2)

            x1 = self.model.layers(wave)
            x2 = self.model.class_encoder(x1)
            features = torch.cat([x1, x2], dim=1)
            out_class = self.model.cl(features).squeeze(dim=2)
            out_time = self.model.tm(features).sigmoid().squeeze() * self.n_stride

            oc = out_class.softmax(dim=1)
            B, C, T = oc.shape
            tgrid = (
                torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride
                + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * self.batchstride
            )
            ot = (out_time + tgrid).reshape(-1)
            oc = oc.permute(0, 2, 1).reshape(-1, C)
        return oc, ot


model = Picker()
model.eval()
input_names = ["wave"]
output_names = ["prob", "time"]
x = torch.randn([500000, 3])
torch.onnx.export(
    model,
    x,
    f"pickers/{MODEL_NAME}.onnx",
    verbose=True,
    dynamic_axes={"wave": {0: "batch"}, "prob": {0: "batch"}, "time": {0: "batch"}},
    input_names=input_names,
    output_names=output_names,
    opset_version=11,
)
