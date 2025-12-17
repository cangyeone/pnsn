import torch
from models.BRNN import BRNN
from jit_picker_base import SlidingWindowPicker


class Picker(SlidingWindowPicker):
    def __init__(self):
        super().__init__(BRNN, ckpt_path="ckpt/china.rnn.pt")


model = Picker()
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/rnn.jit")
x = torch.randn([300000, 3])
y = model(x)
