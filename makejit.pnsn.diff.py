import torch
from models.BRNNPNSN import BRNN
from jit_picker_base import SlidingWindowPicker


class Picker(SlidingWindowPicker):
    def __init__(self):
        super().__init__(BRNN, ckpt_path="ckpt/china.rnn.pnsn.pt", norm="std", diff=True, seqlen=10240, overlap=1024, threshold=0.1, min_gap=300)


model = Picker()
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/pnsn.v1.diff.jit")
x = torch.randn([300000, 3])
y = model(x)
