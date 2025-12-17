import torch
from models.UNet import UNet
from jit_picker_base import SlidingWindowPicker


class Picker(SlidingWindowPicker):
    def __init__(self):
        super().__init__(UNet, ckpt_path="ckpt/china.unet.pt")


model = Picker()
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/unet.jit")
x = torch.randn([300000, 3])
y = model(x)
