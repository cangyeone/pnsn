### 1. Instructions for using the national 100Hz model
All models in this repository are trained on 2009-2019 national seismic network data at 100 Hz. They can be applied directly to continuous three-component waveforms for automatic phase picking.

* Training covers stations within 800 km of the epicenter and includes P/S phases.
* PhaseNet, RNN and LPPN style models have been validated on ChinArray data with RNN recall ≥ 80% on manually labelled sets.
* Accuracy and speed comparisons are shown in `pickers/speed.jpg`.

#### 1.1 Open sourced models
|Model|Size(MB)|P-F1Score|Instrument|Sampling rate|Channel|Max distance|Range|Output phases|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|BRNN|1.9|0.857|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|EQTrasformer|3.1|0.852|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|PhaseNet(UNet)|0.8|0.815|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|LPPN(Large)|2.7|0.813|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|LPPN(Medium)|0.4|0.808|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|LPPN(Tinny)|0.3|0.757|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|UNet++|12|0.798|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|pnsn (**used in the paper**)|1.9|0.781|Broad band, MEMS|100Hz|EHZ|2000km|Global|Pg、Sg、Pn、Sn|
|pnsn.diff (**used in the paper**)|1.9|0.781|Broad band, MEMS|100Hz|EHZ|2000km|Global|Pg、Sg、Pn、Sn|
|tele|1.9|0.800|Broad band|20Hz|EHZ|>3000km|Global|P|

`pickers/pnsn.jit` implements the first inference strategy and `pickers/pnsn.diff.jit` the second; both accept waveforms of arbitrary length.

#### 1.2 TorchScript quick start
The TorchScript models in `pickers/` ship with all post-processing (thresholding and non-maximum suppression) baked into the graph. They expect three-component waveforms resampled to 100 Hz and output `[phase_type, relative_sample, confidence]` for each pick.
```python 
import numpy as np  # Import NumPy for numerical operations
import torch         # Import PyTorch for loading and running the model
import obspy         # Import ObsPy for reading seismic waveform data (install via `pip install obspy`)

mname = "pickers/pnsn.jit"  # Path to the TorchScript seismic phase picking model
device = torch.device("cpu")  # Set the inference device to CPU
sess = torch.jit.load(mname)  # Load the serialized TorchScript model
sess.eval()  # Set the model to evaluation (inference) mode
sess.to(device)  # Move the model to the specified device (CPU)

# Read three-component seismic waveform data from SAC files
st1 = obspy.read("data/waveform/X1.53085.01.BHE.D.20122080726235953.sac")  # East component
st2 = obspy.read("data/waveform/X1.53085.01.BHN.D.20122080726235953.sac")  # North component
st3 = obspy.read("data/waveform/X1.53085.01.BHZ.D.20122080726235953.sac")  # Vertical component
data = [st1[0].data, st2[0].data, st3[0].data]  # Extract waveform arrays from the Trace objects

x = np.stack(data, axis=1).astype(np.float32)  # Stack the 3 components into shape [N, 3] and convert to float32
with torch.no_grad():  # Disable gradient tracking for inference
    x = torch.tensor(x, dtype=torch.float32, device=device)  # Convert NumPy array to PyTorch tensor on CPU
    y = sess(x)  # Run inference through the model
    phase = y.cpu().numpy()  # Convert model output to NumPy array (post-processing done separately)

import matplotlib.pyplot as plt  # Import Matplotlib for plotting

plt.plot(x[:, 2], alpha=0.5)  # Plot the vertical (Z) component waveform

# Loop over each picked phase and draw a vertical line at the picked time
for pha in phase:
    if pha[0]==0:  # Pg phase
        c = "r"  # Red
    elif pha[0]==1:  # Sg phase
        c = "b"  # Blue
    elif pha[0]==2:  # Pn phase
        c = "g"  # Green
    else:  # Sn phase or others
        c = "k"  # Black
    plt.axvline(pha[1], c=c)  # Draw vertical line at phase pick time with corresponding color

plt.show()  # Display the plot

```



#### 1.3 Recommended models
1. If accuracy is most important, prefer the pnsn/pnsn.diff variants (tested on mobile, dense, and fixed global networks).
2. If memory is tight or speed matters, choose LPPN models.
3. For low recall scenarios, lower the confidence threshold to 0.1 (for example `pickers/rnn.01.jit`) or use the Pn/Sn-aware models.
4. When per-sample confidence traces are required, use an ONNX model and handle post-processing externally.

#### 1.4 Pn and Sn phase picking model
1. In order to make the model more universal, we trained a new model using 2000km of manually labeled data.
2. The model is called rnn.pnsn.jit.
3. Based on the RNN model, it can simultaneously pick P, S, Pn, and Sn phases.
4. For other code, please visit the Gitee project address.
5. Due to the imbalanced nature of the data and some missing phase labels leading to low confidence, the confidence threshold is currently set at 0.1.
6. The model can be called by pickers.py for automatic picking by traversing directories directly as described in section 4.1.
7. The data needs to be sampled at 100Hz.
8.The accuracy has not been fully tested yet; only 10,000 waveforms of 102.4 seconds within 2000km from year 2020 were used for testing with results shown in the figure.
9.We found that after high-pass filtering (differentiation), the picking effect for large earthquakes was better; therefore we created a model for picking original + differentiated data as an example: makejit.pnsn.diff.py.Output models are: rnn.origdiff.pnsn.jit
10. The pnsn model was originally released in 2022, and updated `pnsn.jit` and `pnsn.diff.jit` were released in 2025 in our paper. The original accuracy is shown in [pickers/china.pnsn.jpg].


Call in python interface
```python 
import torch 
sess = torch.jit.load("rnn.pnsn.jit")
x = ... # [Any length, 3] 
with torch.no_grad():
    x = torch.tensor(x, dtype=torch.float32, device=device) 
    y = sess(x) 
    phase = y.cpu().numpy()# [Number of phases, 1P, 2S, 3Pn, 4Sn]
```

#### 1.5 Distant Earthquake Picking Model
We provide `tele.rnn.jit` for distant event picking. It outputs distant P/S phases at 20 Hz.

#### 1.6 Environment and data prerequisites
The examples in this repository rely on common scientific Python packages: `torch`, `numpy`, `obspy`, `scipy`, `matplotlib`, and `tqdm` (see the imports in `picker.py`). The picker utilities assume three-component waveforms sampled at 100 Hz with channel names such as `BHE/BHN/BHZ` and file extensions ending in `.mseed` by default (see `config/picker.py`).

### 2. Model Usage Instructions
We provide three types of model files:
1. `.pt` files in the `ckpt` folder, which can be used for transfer learning. Freeze some parameters when adapting to local data.
2. Models for picking any length are located in the `pickers` folder.
   - `.jit` for direct use with PyTorch; post-processing is embedded in the graph and outputs `[phase_type, relative_sample, confidence]` per pick.
   - `.onnx` for use with `onnxruntime`, suitable for edge devices. Use the `post` functions in `picker.onnx.py` or `picker.py` to apply the probability threshold (`a`) and non-maximum suppression window (`b`) to the raw `prob` and `time` outputs.
- `.jit` output format: `[number of phases, phase type + relative arrival time + confidence]`. Phase types: 1:P, 2:S (Pn/Sn models extend this list).
- `.onnx` outputs two tensors: `prob[i]` (per-sample class probabilities, length 3) and `time[i]` (relative sample index). Combine them with post-processing to form picks.
- Example usage of .jit can be found in `picker.jit.py`.
- Example usage of .onnx can be found in `picker.onnx.py`.
  
#### 2.1 Using C Language Version Onnx Model
For C users, `.merge.onnx` files combine the `time` and `prob` outputs into a single array:
```
[ [time length, number of categories, -, -],
  [number of categories, noise probability, P-wave probability, S-wave probability],
  [sample points, noise probability, P-wave probability, S-wave probability],
  ... ]
```
For example programs in C, contact yuziye@cea-igp.ac.cn.

### 3. make onnx and jit files
#### 3.1 Building `.jit` pickers
All TorchScript pickers share the same interface via `jit_picker_base.py::SlidingWindowPicker`. Each `makejit.XXX.py` file simply:
1. Constructs the underlying network with `self.model = UNet()`/`BRNN()`/`EQTransformer()`, etc.
2. Loads a checkpoint whose keys are prefixed with `model.` (legacy checkpoints are also accepted and will be auto-prefixed).
3. Wraps the network with sliding-window preprocessing, softmax, and non-maximum suppression.
4. Saves the scripted model into `pickers/*.jit`.

To rebuild the packaged TorchScript files, run the corresponding script (for example `python makejit.unet.py`, `python makejit.unetpp.py`, `python makejit.rnn.py`, `python makejit.pnsn.py`, or `python makejit.eqt.py`). The output `.jit` files include post-processing, so they return `[phase_type, relative_sample, confidence]` directly when you call `torch.jit.load`.

Key thresholds baked into the picker interface:
```python
time_sel = torch.masked_select(ot, pc > 0.3)  # confidence threshold
selidx = torch.masked_select(selidx, torch.abs(ref - ntime) > 1000)  # NMS window (samples)
```
* `0.3` is the default minimum confidence. Lower it to pick more candidates at the cost of extra false triggers.
* `1000` samples (10 seconds at 100 Hz) enforce a single pick per class within that window. Reduce the window if multiple phases are expected in short succession.

#### 3.2 Building `.onnx` pickers
Use the companion `makeonnx.XXX.py` scripts to export ONNX versions of each network. **The onnx model can use config/picker.py for post-processing as it is outside of the model itself**


### 4. 发布到 PyPI（pip 安装）
如果希望让用户直接 `pip install pnsn`，可以按下面步骤整理并发布到 PyPI：

1. **准备包目录**：在仓库根目录下新增一个 `pnsn/` 目录，并将可复用的 Python 模块（例如 `picker.py`、`picker.onnx.py` 等改名为合法模块名后放入其中），创建空的 `pnsn/__init__.py` 以便 `setuptools` 识别为包。
2. **包含模型资源**：在 `MANIFEST.in`（需要新建）中添加类似 `recursive-include pickers *.jit *.onnx` 以便在打包时附带模型文件；同时在 `pyproject.toml` 中开启 `include-package-data = true`。
3. **添加构建元数据**：创建 `pyproject.toml`，使用 `setuptools.build_meta` 作为构建后端，并补充项目名称、版本、作者、依赖等信息。例如：
   ```toml
   [build-system]
   requires = ["setuptools>=64", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "pnsn"
   version = "0.1.0"
   description = "P/S/Pn/Sn picking toolkit"
   authors = [{ name = "Your Name", email = "you@example.com" }]
   requires-python = ">=3.8"
   dependencies = ["torch", "obspy", "numpy", "scipy", "tqdm"]

   [tool.setuptools]
   include-package-data = true
   packages = ["pnsn"]

   [project.scripts]
   pnsn-picker = "pnsn.picker:main"          # 将 pnsn/picker.py 中的入口函数暴露为 CLI
   pnsn-picker-onnx = "pnsn.picker_onnx:main"  # 将 onnx 版本入口暴露为 CLI
   ```
4. **构建并本地验证**：安装构建工具后生成发布文件，并用 pip 从本地文件验证安装是否成功：
   ```bash
   python -m pip install --upgrade build twine
   python -m build         # 在 dist/ 生成 .whl 和 .tar.gz
   python -m pip install dist/pnsn-0.1.0-py3-none-any.whl
   ```
5. **上传到 PyPI**：在测试仓库 `test.pypi.org` 上先行验证再上传正式 PyPI：
   ```bash
   # 上传到 TestPyPI
   python -m twine upload --repository testpypi dist/*
   # 上传到正式 PyPI（确认版本号未被占用）
   python -m twine upload dist/*
   ```

### 5. Directly picking up continuous data
#### 5.1 Phase picking
Phase picking provides a more convenient way to directly traverse the directory and pick up all phases.
```bash 
python picker.py -i path/to/data -o outputname -m pickers/rnn.jit -d device
```

1. output file name.txt containing all picked phases 
2. output file name.log containing processed data information
3. output file name.err containing problematic data information

The format of the output file is:
```text
#path/to/file
phase name,relative time(s),confident,aboulute time(%Y-%m-%d %H:%M:%S.%f),SNR,AMP,station name,other information
```

`picker.py` exposes the `-i/--input`, `-o/--output`, `-m/--model`, and `-d/--device` arguments (see `if __name__ == "__main__"` in the script) and uses the defaults from `config/picker.py` for details such as channel count (`nchannel=3`), sampling rate (`samplerate=100`), probability threshold for ONNX models (`prob=0.3`), and non-maximum suppression window (`nmslen=1000`).


#### 5.2 Seimic assosication
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
```
SC AXX 00 110.00 38.00 1000.00
```


The structure of the output association file is:
```text
##EVENT,TIME,LAT,LON,DEP##
PHASE,TIME,LAT,LON,TYPE,PROB,STATION,DIST,DELTA,ERROR#
EVENT,2022-04-09 02:28:38.021000,100.6492,25.3660,PICKED_PHASE_TIME_LAT_LON_TYPE_PROB_STATION_DIST_DELTA_ERROR#
PHASE_PICKED_TIME_LAT_LON_TYPE_PROB_STATION_DIST_DELTA_ERROR#
```

### Open Source License
GPLv3

### Related publication
* **Journal:** Journal of Geophysical Research: Machine Learning and Computation (Open Access)
* **Title:** *A Deep Learning Framework for Pg/Sg/Pn/Sn Phase Picking and Its Nationwide Implementation in Chinese Mainland*
* **DOI:** 10.1029/2025JH000944
* **Status:** In Production
