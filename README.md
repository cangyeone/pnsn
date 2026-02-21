<p align="center">
  <img src="logo.png" alt="SeismicXM logo"/>
</p>






# SeismicX-PnSn: A Deep Learning Framework for Pg/Sg/Pn/Sn Phase Picking and Its Nationwide Implementation in Chinese Mainland

**Code for:** 
* **Title:** *A Deep Learning Framework for Pg/Sg/Pn/Sn Phase Picking and Its Nationwide Implementation in Mainland China* 
* **Authors:** Yuqi Cai, Ziye Yu, et al. ([yuziye@cea-igp.ac.cn](mailto:yuziye@cea-igp.ac.cn))
* **DOI:** https://doi.org/10.1029/2025JH000944

All models in this repository are trained on **2009–2019** national seismic network data at **100 Hz**. They are designed for **direct inference on continuous three-component waveforms** (E/N/Z) for automatic phase picking.

Key notes:

* Training primarily covers stations within ~800 km and includes local/regional **P/S** phases (Pg/Sg).
* PhaseNet / RNN / LPPN-style models have been validated on ChinArray data with RNN recall ≥ 80% on manually labeled sets.
* Accuracy and speed comparisons are shown in `pickers/speed.jpg`.
## 1. Introduction
### 1.0 About the pnsn model family

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

### 1.2 TorchScript quick start 

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

## 2. Model Usage Instructions 

We provide three types of model files: 
1. .pt files in the ckpt folder, which can be used for transfer learning. Freeze some parameters when adapting to local data.
2. Models for picking any length are located in the pickers folder.
   - .jit for direct use with PyTorch; post-processing is embedded in the graph and outputs [phase_type, relative_sample, confidence] per pick.
   - .onnx for use with onnxruntime, suitable for edge devices.

Use the post functions in picker.onnx.py or picker.py to apply the probability threshold (a) and non-maximum suppression window (b) to the raw prob and time outputs. 
   - .jit output format: [[Type of phases, relative arrival time, confidence], ...]. Phase types: 0:Pg, 1:Sg, 2:Pn, 3:Sn (Pn/Sn models extend this list).
   - .onnx outputs two tensors: prob[i] (per-sample class probabilities, length 3 (indicate Noise, Pg, Sg) or 5 (indicate Noise, Pg, Sg, Pn, Sn)) and time[i] (relative sample index). Combine them with post-processing to form picks.
   - Example usage of .jit can be found in picker.jit.py.
   - Example usage of .onnx can be found in picker.onnx.py. 

**When running the phase NMS algorithm with TorchScript (.jit) models, performance bottlenecks may occur; in such cases, it is recommended to use ONNX-based inference instead. A reference implementation is provided in `picker.onnx.py`, and `picker.py` can be configured to load and run a `.onnx` model directly.**

   
### 2.1 Using C Language Version Onnx Model 

For C users, .merge.onnx files combine the time and prob outputs into a single array:
[ [time length, number of categories, -, -],
  [number of categories, noise probability, P-wave probability, S-wave probability],
  [sample points, noise probability, P-wave probability, S-wave probability],
  ... ]
For example programs in C, contact yuziye@cea-igp.ac.cn. 

### 2.2 Building .jit pickers 

All TorchScript pickers share the same interface via jit_picker_base.py::SlidingWindowPicker. 
Each makejit.XXX.py file simply: 

1. Constructs the underlying network with self.model = UNet()/BRNN()/EQTransformer(), etc.
2. Loads a checkpoint whose keys are prefixed with model. (legacy checkpoints are also accepted and will be auto-prefixed).
3. Wraps the network with sliding-window preprocessing, softmax, and non-maximum suppression.
4. Saves the scripted model into pickers/*.jit.
  
To rebuild the TorchScript files, run the corresponding script (for example python makejit.unet.py, python makejit.unetpp.py, python makejit.rnn.py, python makejit.pnsn.py, or python makejit.eqt.py). 
The output .jit files include post-processing, so they return [phase_type, relative_sample, confidence] directly when you call torch.jit.load. Key thresholds baked into the picker interface:

```python
time_sel = torch.masked_select(ot, pc > 0.3)  # confidence threshold
selidx = torch.masked_select(selidx, torch.abs(ref - ntime) > 1000)  # NMS window (samples)
```

* 0.3 is the default minimum confidence. Lower it to pick more candidates at the cost of extra false triggers.
* 1000 samples (10 seconds at 100 Hz) enforce a single pick per class within that window.

Reduce the window if multiple phases are expected in short succession. 

### 2.3 Building .onnx pickers 

All ONNX pickers share the OnnxSlidingWindowPicker interface defined in onnx_picker_base.py. 
To regenerate the exported ONNX files: 

1. Run the corresponding script (for example python makeonnx.unet.py, python makeonnx.unetpp.py, python makeonnx.rnn.py, python makeonnx.pnsn.py, or python makeonnx.eqt.py).
2. Each script builds the model (self.model = UNet()/BRNN()/EQTransformer(), etc.), loads checkpoints (auto-prefixing with model. when needed), and wraps it with the shared sliding-window preprocessing.
3. Post-processing (probability threshold and NMS) remains outside the ONNX graph; reuse config/picker.py together with the post helpers in picker.onnx.py or picker.py when running inference.
  
**The onnx model can use picker.py for post-processing as it is outside of the model itself** 

## 3. Directly picking up continuous data 

### 3.1 Phase picking 

Phase picking provides a more convenient way to directly traverse the directory and pick up all phases.
```bash
python picker.py -i path/to/data -o outputname -m pickers/rnn.jit -d device
```

1. output file name.txt containing all picked phases
2. output file name.log containing processed data information
3. output file name.err containing problematic data information The format of the output file is:

```text
#path/to/file
phase name,relative time(s),confident,aboulute time(%Y-%m-%d %H:%M:%S.%f),SNR,AMP,station name,other information
```

picker.py exposes the -i/--input, -o/--output, -m/--model, and -d/--device arguments (see if __name__ == "__main__" in the script) and uses the defaults from config/picker.py for details such as channel count (nchannel=3), sampling rate (samplerate=100), probability threshold for ONNX models (prob=0.3), and non-maximum suppression window (nmslen=1000). 


#### 3.1.1 Configuration and File Organization Requirements

Before running the picker, you **must** modify the configuration file `config/picker.py` to match your waveform file format and data organization.


#### 3.1.2 Three-Component Requirement (Critical)

**Each station must be represented by exactly three component files within the same directory.**

* The picker assumes **one station = one 3-component set (E/N/Z or equivalent)**.
* If more than three component files exist for the same station in a directory, **the extra files will be ignored and NOT processed**.
* If fewer than three components are found, **the station will be skipped entirely**.

This design is intentional and ensures strict consistency for three-component phase picking and polarity analysis.


#### 3.1.3 How Three-Component Correspondence Is Determined (Very Important)

Many users organize their data by placing **all time segments of a station into the same directory**.
**This is NOT supported by default.**

The correspondence between three components is determined **only** by:

```python
namekeyindex = [0, 1]
```

That is:

* The picker groups files **solely based on the filename fields specified by `namekeyindex`**
  (e.g., `NET` and `STATION`).
* **Time information in the filename is ignored** unless it is explicitly included in `namekeyindex`.

As a result:

* If a directory contains multiple time segments for the same station (e.g., daily or hourly files),
* and those files share the same `NET.STATION` identifiers,
* **they will all be treated as belonging to the same station**, causing ambiguity.

➡ **Such files will NOT be processed unless the time dimension is explicitly added to `namekeyindex`.**


#### 3.1.4 Recommended File Naming Convention

Example filename:

```
SC.A0801.40.EIE.D.20221400520064953.sac
```

Expected format:

```
NET.STATION.LOC.CHANNEL.OTHERS.mseed
```

Relevant configuration:

```python
namekeyindex = [0, 1]   # NET, STATION
channelindex = 3        # CHANNEL
```

If you want to process **multiple time segments per station**, you must:

* Either separate them into different directories, **or**
* Include a time-related field (e.g., date or start time) in `namekeyindex`.


#### 3.1.5 Channel Name Matching

The picker supports the following three-component channel groups:

```python
chnames = [
    ["BHE", "BHN", "BHZ"],
    ["SHE", "SHN", "SHZ"],
    ["HHE", "HHN", "HHZ"],
    ["EIE", "EIN", "EIZ"],
    ["HNN", "HNE", "HNZ"],
    ["E", "N", "Z"],
]
```

Notes:

* **All possible channel name combinations must be listed**.
* The **Z component must always be the last entry**, as it is required for first-motion polarity calculation.
* You need to **modify  `channelindex`** in config file to tell which part in name indicate the channel. 


### 3.2 Seimic assosication

The goal of seismic association is to determine the number, location, and timing information of earthquakes from the phase picking results. Currently, there are 3 association algorithms provided: 
1. REAL methods [reallinker.py]
2. LPPN methods [fastlinker.py]
3. GaMMA methods [gammalinker.py]
  
Both models take the picking results as input.
```bash
python fastlinker.py -i phase_picking_results.txt -o output_file_name.txt -s station_directory
```

The format of the station file is:
```text
network station LOC longitude latitude elevation(m)
```

For example:
SC AXX 00 110.00 38.00 1000.00

The structure of the output association file is:
```text
#EVENT,TIME,LAT,LON,DEP
PHASE,TIME,LAT,LON,TYPE,PROB,STATION,DIST,DELTA,ERROR#
EVENT,2022-04-09 02:28:38.021000,100.6492,25.3660,PICKED_PHASE_TIME_LAT_LON_TYPE_PROB_STATION_DIST_DELTA_ERROR#
PHASE_PICKED_TIME_LAT_LON_TYPE_PROB_STATION_DIST_DELTA_ERROR#
```

## 4. Python-based REAL Earthquake Association Engine 

This project provides a **Python implementation of the REAL (Rapid Earthquake Association and Location) algorithm** for automated earthquake phase association and preliminary event location. The implementation preserves the core logic, decision criteria, and workflow of the original C-based REAL algorithm, while introducing improved modularity, flexible data interfaces, and high-performance acceleration through modern Python tooling. The algorithm follows the canonical REAL workflow of **P-phase initiation → grid-based association → multi-phase consistency scoring → event confirmation**. All P-phase picks are processed in chronological order using a heap-based scheduler. Each candidate trigger initiates a local three-dimensional grid search (latitude, longitude, depth), where theoretical P- and S-wave travel times are computed under a homogeneous velocity model. Observed picks falling within adaptive time windows are associated, and candidate sources are evaluated using phase count thresholds and robust origin-time scatter metrics. 

### Key Features 
* **High-performance numerical core** The computationally intensive grid evaluation and phase matching are implemented using Numba JIT compilation, achieving performance comparable to the original C REAL code while maintaining Python-level readability.
* **Flexible pick input formats** The engine supports both traditional per-station pick files and modern single-file, multi-station pick formats that include absolute timestamps and confidence scores. All inputs are internally mapped to fixed-size NumPy arrays for efficient vectorized and JIT-compiled processing.
* **Confidence-aware association** Pick confidence values are preserved throughout the association process and propagated to event-phase outputs, enabling downstream quality control, weighted relocation, or uncertainty analysis.
* **Strict consistency with C-REAL criteria** Core constraints such as minimum P/S counts, P+S totals, same-station P–S pairing, dtps limits, and the ispeed phase-removal mechanism are implemented to closely match the behavior of the original REAL algorithm.
* **Event–phase level output** For each detected event, the engine outputs both event-level parameters (origin time, hypocenter, scatter) and detailed phase associations, including pick time, residual, source–receiver distance, confidence, and station identifier.

Overall, this Python-based REAL implementation balances **algorithmic fidelity, computational efficiency, and extensibility**, making it suitable for both research-oriented seismic analysis and large-scale automated earthquake catalog production. 

### 4.1 Usage 

### 4.2 Input Data 

Station file The station file follows the original REAL format:
NET STA COMP STLO STLA ELEV
where elevation is given in meters. Example:
YN YSW03 BHZ 102.345 24.567 1850

Pick data Two pick input modes are supported: 

1. **Per-station files (C-REAL compatible)** Each station has separate P and S files:
NET.STA.P.txt
NET.STA.S.txt
with columns:
trigger_time  weight  amplitude
2. **Single-file multi-station picks (recommended)** A single CSV-style file containing all picks:
#data/xxx/YN.YSW03.00.mseed
Pg,32640.160,0.936,2021-05-21 09:04:00.165000,...,YN.YSW03.00,...
Required fields: * phase name (e.g., Pg, Pn, Sg) * relative trigger time (seconds) * confidence * absolute time string * station identifier (NET.STA.LOC) ###

### 4.3 Basic Workflow
```python
from real import (
    read_station_txt,
    load_all_picks_from_singlefile_v2,
    FastREAL
)
```


1. Load stations

stla, stlo, elev, net, sta = read_station_txt("stations.txt")


2. Load picks

MAXTIME = 2.7e6
NNps = 20000

ptrig, strig, pconf, sconf, pabs, sabs = load_all_picks_from_singlefile_v2(
    pick_file="picks.csv",
    net=net,
    sta=sta,
    max_n=NNps,
    max_time=MAXTIME,
    min_conf=0.0,
)


3. Initialize REAL engine

real = FastREAL(
    stla, stlo, elev,
    ptrig, strig,
    pabs, sabs, pconf, sconf,
    lat_center_deg=25.0,
    rx_deg=1.0, rh_km=30.0,
    dx_deg=0.05, dh_km=2.0,
    tint_sec=10.0,
    vp0=6.0, vs0=3.5,
    s_vp0=6.0, s_vs0=3.5,
    np0=6, ns0=4, nps0=10, npsboth0=2,
    std0=1.0,
    dtps=2.0,
    nrt=2.0,
    gcarc0_deg=3.0,
    ispeed=True,
    max_time=MAXTIME,
    net=net,
    sta=sta,
)


4. Run association

events, event_phases = real.run(
    latref0_init=None,
    lonref0_init=None,
    max_events=100000
)

### 4.4 Output 

Each detected event is returned as:
(origin_time, latitude, longitude, depth,
 scatter, P_count, S_count, P+S_count, PS_both)
Associated phase information is stored per event as:
(phase, pick_time_str, dt, distance_km, confidence, residual, station_id)
Optionally, results can be written to disk in a REAL-compatible text format using:

```python
write_events_real_format(
    out_path="events.txt",
    events=events,
    real=real,
    pabs=pabs,
    sabs=sabs,
    pconf=pconf,
    sconf=sconf,
    per_event_phase_rows=event_phases,
)
```

### 4.5 Notes 
* The current implementation assumes a **homogeneous velocity model**; extension to layered or 3-D models can be achieved by replacing the travel-time kernel.
* Absolute times are treated as **naive timestamps** and used only for relative timing and output reconstruction.
* For large station counts or dense pick sets, enabling Numba (NUMBA_OK = True) is strongly recommended.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) for non-commercial academic and private research use.
Permitted Use (Academic / Private Research)
- Use, study, and modify the code for non-commercial academic research or personal research purposes
- Redistribution of modified or unmodified versions must comply with GPL-3.0
- Proper citation of the author and this repository is required in any academic publication or derivative work
- This code constitutes core methodology and must be acknowledged as such

Commercial Use
Any commercial use, including but not limited to:
- Commercial research and development
- Product integration or deployment
- Paid services, proprietary systems, or closed-source redistribution

is NOT permitted under this license.
For commercial licensing, redistribution, or integration into proprietary systems,
please contact the corresponding author to obtain explicit written permission and discuss licensing terms.

## Contact
Yuqi Cai: caiyuqiming@foxmail.com

