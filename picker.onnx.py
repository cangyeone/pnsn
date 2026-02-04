import numpy as np
import onnxruntime as ort
import obspy # pip install obspy 

import numpy as np
import heapq
from bisect import bisect_left

def post(prob, time, prob_thresh=0.1, nms_win=200):
    """
    a: 概率阈值
    b: NMS 去重时间间隔（同一类内，已选点±b范围内的候选点会被抑制）
    使用最小堆（heapq）按 score 从大到小弹出（用 -score 实现）。
    """
    output = []
    t, c = prob.shape

    for itr in range(c - 1):
        pc = prob[:, itr + 1]

        mask = pc > prob_thresh
        if not np.any(mask):
            continue

        time_sel = time[mask]
        score_sel = pc[mask]

        # 最小堆：存 (-score, time, idx_in_sel)
        heap = [(-float(s), float(ts), i) for i, (s, ts) in enumerate(zip(score_sel, time_sel))]
        heapq.heapify(heap)

        # 已接受 pick 的 time（保持有序，便于用 bisect 快速检查最近邻）
        accepted_times = []
        accepted_idx = []

        while heap:
            neg_s, ts, i = heapq.heappop(heap)
            s = -neg_s

            # 检查 ts 是否与已选时间点冲突（只需看有序列表的前后邻居）
            pos = bisect_left(accepted_times, ts)

            conflict = False
            if pos > 0 and abs(ts - accepted_times[pos - 1]) <= nms_win:
                conflict = True
            if pos < len(accepted_times) and abs(accepted_times[pos] - ts) <= nms_win:
                conflict = True

            if conflict:
                continue

            accepted_times.insert(pos, ts)
            accepted_idx.append(i)

        if len(accepted_idx) == 0:
            continue

        p_time = time_sel[accepted_idx]
        p_prob = score_sel[accepted_idx]
        p_type = np.full_like(p_time, itr, dtype=np.float32)

        y = np.stack([p_type, p_time.astype(np.float32), p_prob.astype(np.float32)], axis=1)
        output.append(y)

    if len(output) == 0:
        return []

    return np.concatenate(output, axis=0)

        
mname = "pickers/pnsn.v1.onnx" # 其他onnx均可
sess = ort.InferenceSession(mname, providers=['CPUExecutionProvider'])#使用pickers中的onnx文件

# 读取数据
st1 = obspy.read("data/waveform/X1.53085.01.BHE.D.20122080726235953.sac")
st2 = obspy.read("data/waveform/X1.53085.01.BHN.D.20122080726235953.sac")
st3 = obspy.read("data/waveform/X1.53085.01.BHZ.D.20122080726235953.sac")
data = [st1[0].data, st2[0].data, st3[0].data] 
# 任意长度数据均可
# 数据不需要滤波、预处理、归一化等操作
x = np.stack(data, axis=1).astype(np.float32)[:] #[N, 3]->一天 [8640000]100Hz 
# 直接运行即可
prob, time = sess.run(["prob", "time"], {"wave":x})
phase = post(prob, time, 0.2, 200)
import matplotlib.pyplot as plt 
plt.plot(x[:, 2], alpha=0.5) 
plt.scatter(time, prob[:, 1]*np.max(x[:, 2]), c="r")
plt.scatter(time, prob[:, 2]*np.max(x[:, 2]), c="b")
for pha in phase:
    if pha[0]==0:
        c = "r" 
    else:
        c = "b"
    plt.axvline(pha[1], c=c)
plt.show()
