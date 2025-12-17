import torch
from models.EQTransformer import EQTransformer
from jit_picker_base import SlidingWindowPicker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained = EQTransformer.from_pretrained("stead").to(device)
pretrained_state = {k: v.cpu() for k, v in pretrained.state_dict().items()}


class Picker(SlidingWindowPicker):
    def __init__(self):
        super().__init__(EQTransformer, state_dict=pretrained_state)


model = Picker()
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/eqtransformer.stead.jit")
x = torch.randn([300000, 3])
y = model(x)
