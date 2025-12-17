import torch
from models.UNetPlusPlus import UNetpp
from jit_picker_base import SlidingWindowPicker


class Picker(SlidingWindowPicker):
    def __init__(self):
        super().__init__(UNetpp, ckpt_path="ckpt/china.unetpp.pt")


model = Picker()
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/unetpp.jit")
x = torch.randn([300000, 3])
y = model(x)
