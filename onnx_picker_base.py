import torch
import torch.nn as nn


class OnnxSlidingWindowPicker(nn.Module):
    """
    Wrapper to standardize ONNX picker interfaces.

    Each picker stores its underlying network in ``self.model`` and handles
    sliding-window preprocessing plus checkpoint loading. Checkpoints are
    expected to prefix parameters with ``model.``; if the prefix is missing,
    it is added automatically for backward compatibility.
    """

    def __init__(
        self,
        model_ctor,
        ckpt_path=None,
        *,
        state_dict=None,
        seqlen=6144,
        overlap=256,
        use_softmax=True,
    ):
        super().__init__()
        self.model = model_ctor()
        self.n_stride = 1
        self.seqlen = seqlen
        self.batchstride = seqlen - overlap
        self.use_softmax = use_softmax

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
            wave, batchlen = self.window_and_normalize(x, device)

            logits = self.model(wave)
            if logits.dim() == 4:
                logits = logits.squeeze(dim=3)
            if self.use_softmax and logits.shape[1] > 1:
                logits = logits.softmax(dim=1)

            B, C, T = logits.shape
            tgrid = (
                torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride
                + torch.arange(0, batchlen, 1, device=device).unsqueeze(1)
                * self.batchstride
            )
            oc = logits.permute(0, 2, 1).reshape(-1, C)
            ot = tgrid.squeeze().reshape(-1)
        return oc, ot

    def window_and_normalize(self, x, device):
        T, _ = x.shape
        batchlen = torch.ceil(torch.tensor(T / self.batchstride).to(device))
        idx = (
            torch.arange(0, self.seqlen, 1, device=device).unsqueeze(0)
            + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * self.batchstride
        )
        idx = idx.clamp(min=0, max=T - 1).long()
        wave = x.to(device)[idx, :]
        wave = wave.permute(0, 2, 1)
        wave -= torch.mean(wave, dim=2, keepdim=True)
        maxv, _ = torch.max(torch.abs(wave), dim=2, keepdim=True)
        wave /= (maxv + 1e-6)
        return wave, batchlen
