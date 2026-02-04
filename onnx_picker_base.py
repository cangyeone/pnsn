import torch
import torch.nn as nn



class OnnxSlidingWindowPicker(nn.Module):
    """
    Wrapper to standardize JIT picker interfaces.

    Each picker holds its underlying network in ``self.model``. The checkpoint
    is expected to use keys prefixed with ``model.``; legacy checkpoints without
    the prefix are supported by automatically adding it during load.
    """
    __annotations__ = {}   # Testing for torchscript
    def __init__(self, model_ctor, ckpt_path=None, *, norm="std", diff=False, state_dict=None,
                 seqlen=6144, overlap=256):
        super().__init__()
        self.model = model_ctor()
        self.n_stride = 1
        self.seqlen = seqlen
        self.batchstride = seqlen - overlap
        self.diff = diff 
        self.norm = norm # "std" or "max"
        if state_dict is None:
            if ckpt_path is None:
                raise ValueError("Either ckpt_path or state_dict must be provided")
            state_dict = torch.load(ckpt_path, map_location="cpu")
        if not any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        device = x.device
        with torch.no_grad():
            T, _ = x.shape
            batchlen = torch.ceil(torch.tensor(T / self.batchstride).to(device))
            idx = (
                torch.arange(0, self.seqlen, 1, device=device).unsqueeze(0)
                + torch.arange(0, batchlen, 1, device=device).unsqueeze(1)
                * self.batchstride
            )
            idx = idx.clamp(min=0, max=T - 2).long()

            
            wave = x.to(device)[idx, :]
            wave = wave.permute(0, 2, 1)
            wave -= torch.mean(wave, dim=2, keepdim=True)
            if self.norm=="std":
                maxv = torch.std(wave, dim=2, keepdim=True)
            else:
                maxv, _ = torch.max(torch.abs(wave), dim=2, keepdim=True)
            wave /= (maxv + 1e-6)

            logits = self.model(wave)
            if logits.dim() == 4:
                logits = logits.squeeze(dim=3)
            #if logits.shape[1] > 1:
            #    logits = logits.softmax(dim=1)

            B, C, T = logits.shape
            tgrid = (
                torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride
                + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * self.batchstride
            )
            oc = logits.permute(0, 2, 1).reshape(-1, C)
            ot = tgrid.squeeze().reshape(-1)

            if self.diff:
                # 2nd pass for diff
                wave = (x[1:]-x[:-1])[idx, :]
                wave = wave.permute(0, 2, 1)
                wave -= torch.mean(wave, dim=2, keepdim=True)
                if self.norm=="std":
                    maxv = torch.std(wave, dim=2, keepdim=True)
                else:
                    maxv, _ = torch.max(torch.abs(wave), dim=2, keepdim=True)
                #max, maxidx = torch.max(torch.abs(wave), dim=2, keepdim=True) 
                wave /= (maxv + 1e-6)  
                logits = self.model(wave)
                #oc2 = y.softmax(dim=1)
                oc2 = logits.permute(0, 2, 1).reshape(-1, C)
                oc = torch.cat([oc, oc2], dim=0)
                ot = torch.cat([ot, ot], dim=0) 
            
        return oc, ot 
