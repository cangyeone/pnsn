import torch
from models.EQT import EQTransformer
from jit_picker_base import SlidingWindowPicker


class Picker(SlidingWindowPicker):
    def __init__(self):
        super().__init__(EQTransformer, ckpt_path="ckpt/china.eqt.pt")


model = Picker()
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/eqt.jit")
x = torch.randn([300000, 3])
y = model(x)
