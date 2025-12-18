<p align="center">
  <img src="logo.png" alt="SeismicXM logo"/>
</p>






## SeismicX-PnSn: A Deep Learning Framework for Pg/Sg/Pn/Sn Phase Picking and Its Nationwide Implementation in Chinese Mainland

**Code for:** 
* **Title:** *A Deep Learning Framework for Pg/Sg/Pn/Sn Phase Picking and Its Nationwide Implementation in Mainland China* 
* **Authors:** Yuqi Cai, Ziye Yu, et al. ([yuziye@cea-igp.ac.cn](mailto:yuziye@cea-igp.ac.cn)) 

All models in this repository are trained on **2009–2019** national seismic network data at **100 Hz**. They are designed for **direct inference on continuous three-component waveforms** (E/N/Z) for automatic phase picking.

Key notes:

* Training primarily covers stations within ~800 km and includes local/regional **P/S** phases (Pg/Sg).
* PhaseNet / RNN / LPPN-style models have been validated on ChinArray data with RNN recall ≥ 80% on manually labeled sets.
* Accuracy and speed comparisons are shown in `pickers/speed.jpg`.

### 1.0 About the pnsn model family (recommended)

The **pnsn** models are designed to detect **Pg, Sg, Pn, and Sn** on continuous streams using **long windows** (≈102.4 s) and sliding inference, which improves robustness and reduces operational false triggers compared with short-window pickers. 

We provide two major generations:

* **`pickers/pnsn.v1.jit`**: the **first engineering version** that has been used in production/engineering workflows.
* **`pickers/pnsn.v3.jit`** and **`pickers/pnsn.diff.v3.jit`**: the **paper (v3) models** used in the manuscript. 

Two inference strategies (v3):

* **raw**: `pickers/pnsn.v3.jit`
* **raw + first-difference (high-pass by differentiation)**: `pickers/pnsn.diff.v3.jit`
  Both accept waveforms of arbitrary length and include post-processing (thresholding + NMS) inside the TorchScript graph.

### 1.1 Open-sourced models

| Model                     | Size (MB) | P-F1Score | Instrument      | Sampling rate | Channels | Max distance | Range  | Output phases  |
| ------------------------- | --------: | --------: | --------------- | ------------: | -------: | -----------: | ------ | -------------- |
| BRNN                      |       1.9 |     0.857 | Broadband       |        100 Hz |       3C |       300 km | Global | Pg, Sg         |
| EQTransformer             |       3.1 |     0.852 | Broadband       |        100 Hz |       3C |       300 km | Global | Pg, Sg         |
| PhaseNet (UNet)           |       0.8 |     0.815 | Broadband       |        100 Hz |       3C |       300 km | Global | Pg, Sg         |
| LPPN (Large)              |       2.7 |     0.813 | Broadband       |        100 Hz |       3C |       300 km | Global | Pg, Sg         |
| LPPN (Medium)             |       0.4 |     0.808 | Broadband       |        100 Hz |       3C |       300 km | Global | Pg, Sg         |
| LPPN (Tiny)               |       0.3 |     0.757 | Broadband       |        100 Hz |       3C |       300 km | Global | Pg, Sg         |
| UNet++                    |        12 |     0.798 | Broadband       |        100 Hz |       3C |       300 km | Global | Pg, Sg         |
| **pnsn.v1 (engineering)** |      ~1.9 |         – | Broadband, MEMS |        100 Hz |       3C |     ~2000 km | Global | Pg, Sg, Pn, Sn |
| **pnsn.v3 (paper)**       |      ~1.9 |     0.781 | Broadband, MEMS |        100 Hz |       3C |     ~2000 km | Global | Pg, Sg, Pn, Sn |
| **pnsn.diff.v3 (paper)**  |      ~1.9 |     0.781 | Broadband, MEMS |        100 Hz |       3C |     ~2000 km | Global | Pg, Sg, Pn, Sn |
| tele                      |      ~1.9 |     0.800 | Broadband       |         20 Hz |       3C |     >3000 km | Global | P              |


**Important update (deployment recommendation):**

* **Teleseismic/distant events can also be picked using the pnsn models** (especially `pnsn.v3` / `pnsn.diff.v3`) in a unified workflow.
* The standalone **`tele.jit` is not recommended** in practice because its performance is not as stable as using pnsn directly on continuous streams (your engineering experience).

### 1.2 TorchScript quick start (recommended)

TorchScript models under `pickers/` include post-processing (confidence thresholding + NMS) and output:

* `[[phase_type, relative_sample, confidence], ...]`

Use the **paper model** by default:

```python
import numpy as np
import torch
import obspy
import matplotlib.pyplot as plt

mname = "pickers/pnsn.v3.jit"  # paper model (v3); use pnsn.v1.jit for legacy engineering pipelines
device = torch.device("cpu")

sess = torch.jit.load(mname).to(device).eval()

stE = obspy.read("data/waveform/XXX.BHE.sac")[0]
stN = obspy.read("data/waveform/XXX.BHN.sac")[0]
stZ = obspy.read("data/waveform/XXX.BHZ.sac")[0]

x = np.stack([stE.data, stN.data, stZ.data], axis=1).astype(np.float32)  # [N, 3]

with torch.no_grad():
    picks = sess(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()

plt.plot(x[:, 2], alpha=0.5)
for pha, idx, conf in picks:
    pha = int(pha)
    c = {0:"r", 1:"b", 2:"g", 3:"k"}.get(pha, "k")  # 0 Pg, 1 Sg, 2 Pn, 3 Sn
    plt.axvline(idx, c=c, alpha=0.8)
plt.show()
```

### 1.3 Recommended models

1. **Best overall (paper + deployment):** `pnsn.v3.jit` and `pnsn.diff.v3.jit` (mobile / dense / fixed networks; Pg/Sg/Pn/Sn).
2. **Legacy engineering compatibility:** use `pnsn.v1.jit` if you need exact alignment with the original production model behavior.
3. **Speed / small memory:** choose LPPN variants.
4. **Low-recall scenarios:** lower the confidence threshold (e.g., to 0.1) and rely on downstream association/QC to control false positives.
5. **Need per-sample probability traces:** use ONNX and apply post-processing externally.

### 1.4 Distant / teleseismic usage

Although a `tele.jit` model is provided, **we recommend using the pnsn model family for distant/teleseismic records as well**, to keep a single unified picker and avoid inconsistent behavior between local and distant pipelines (based on operational experience).

### 1.5 Environment and data prerequisites

Dependencies: `torch`, `numpy`, `obspy`, `scipy`, `matplotlib`, `tqdm`.

Input assumptions:

* 3-component waveform (E/N/Z), typically resampled to **100 Hz** for pnsn/most models.
* Typical channel naming: `BHE/BHN/BHZ`.
* CLI defaults for continuous data traversal are defined in `config/picker.py`.



## License

* **Research and academic use:** released under **GPLv3**.
* **Commercial use / integration / redistribution:** please contact the corresponding author to obtain permission and discuss licensing terms (email below). 



