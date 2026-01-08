import os
import heapq
import numpy as np

try:
    from numba import njit, prange
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco
    def prange(x): return range(x)


from collections import defaultdict

def read_pickfile_grouped(
    path: str,
    max_time: float,
    min_conf: float = 0.0,
    phase_map: dict = None,
):
    """
    读取你给的格式：
      #<data_path>
      Pg,32640.160,0.936,2021-...,23.4,895.9,YN.YSW03.00,N,0.000

    返回：
      p_by_sta: dict[sta] -> list[(t, conf, amp)]
      s_by_sta: dict[sta] -> list[(t, conf, amp)]
    说明：
      - t 使用“相对时间(秒)”（第2列）
      - conf 使用“置信度”（第3列） -> 可映射为 weight
      - amp 这里用“前后200个采样点振幅均值”（第6列）当 amp（你也可换成 SNR）
      - sta 使用“台站”（第7列），如 'YN.YSW03.00'
    """
    if phase_map is None:
        # 你可以按需扩展：Pn/Pg/P/Sn/Sg 等
        phase_map = {
            "P": "P", "Pg": "P", "Pn": "P", "P1": "P",
            "S": "S", "Sg": "S", "Sn": "S", "S1": "S",
        }

    p_by_sta = defaultdict(list)
    s_by_sta = defaultdict(list)

    cur_data_path = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                cur_data_path = line[1:].strip()
                continue

            a = [x.strip() for x in line.split(",")]
            if len(a) < 7:
                continue

            ph = a[0]
            if ph not in phase_map:
                continue
            phs = phase_map[ph]  # 'P' or 'S'

            try:
                t = float(a[1])
                conf = float(a[2])
            except Exception:
                continue

            if t > max_time:
                continue
            if conf < min_conf:
                continue

            # amp：用第6列“前后200点振幅均值”（如果不存在则 0）
            amp = 0.0
            if len(a) >= 6:
                try:
                    amp = float(a[5])
                except Exception:
                    amp = 0.0

            sta = a[6]  # 'YN.YSW03.00'
            if phs == "P":
                p_by_sta[sta].append((t, conf, amp))
            else:
                s_by_sta[sta].append((t, conf, amp))

    return p_by_sta, s_by_sta

def build_pick_arrays_from_grouped(
    net: list, sta: list,
    p_by_sta: dict, s_by_sta: dict,
    max_n: int, max_time: float,
    min_conf: float = 0.0,
):
    """
    输出：
      ptrig0, strig0, pW, sW, pA, sA  (shape=(Nst, max_n))
    约定：
      weight = conf
      amp    = amp
    """
    nst = len(net)
    ptrig0 = np.full((nst, max_n), INF, np.float32)
    strig0 = np.full((nst, max_n), INF, np.float32)
    pW = np.zeros((nst, max_n), np.float32)
    sW = np.zeros((nst, max_n), np.float32)
    pA = np.zeros((nst, max_n), np.float32)
    sA = np.zeros((nst, max_n), np.float32)

    # 建立 station key：兼容 'NET.STA' 与 'NET.STA.LOC'
    # station.txt 里是 net[i], sta[i]，如 'YN', 'YSW03'
    # pick 里是 'YN.YSW03.00'（含 loc）
    for i in range(nst):
        key2 = f"{net[i]}.{sta[i]}"     # 'YN.YSW03'
        # 在 dict 里找所有以 key2 开头的（可能有多个 loc），合并
        p_list = []
        s_list = []
        for k, v in p_by_sta.items():
            if k.startswith(key2 + ".") or k == key2:
                p_list.extend(v)
        for k, v in s_by_sta.items():
            if k.startswith(key2 + ".") or k == key2:
                s_list.extend(v)

        # 排序并截断
        if p_list:
            p_list.sort(key=lambda x: x[0])
            n = min(len(p_list), max_n)
            for j in range(n):
                t, w, a = p_list[j]
                if t <= max_time and w >= min_conf:
                    ptrig0[i, j] = np.float32(t)
                    pW[i, j] = np.float32(w)
                    pA[i, j] = np.float32(a)

        if s_list:
            s_list.sort(key=lambda x: x[0])
            n = min(len(s_list), max_n)
            for j in range(n):
                t, w, a = s_list[j]
                if t <= max_time and w >= min_conf:
                    strig0[i, j] = np.float32(t)
                    sW[i, j] = np.float32(w)
                    sA[i, j] = np.float32(a)

    return ptrig0, strig0, pW, sW, pA, sA


INF = 1.0e8
DEG2KM = 111.19


# ----------------------------
# IO: station / picks
# ----------------------------
def read_station_txt(path: str):
    """
    station file format (same as C):
      stlo stla net sta comp elev
    """
    stlo = []
    stla = []
    elev = []
    net = []
    sta = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = line.split()
            stlo.append(float(a[3]))
            stla.append(float(a[4]))
            net.append(a[0])
            sta.append(a[1])
            elev.append(float(a[5])/1000.0)
    stlo = np.asarray(stlo, np.float32)
    stla = np.asarray(stla, np.float32)
    elev = np.asarray(elev, np.float32)
    return stla, stlo, elev, net, sta


def read_pick_triplets(path: str, max_n: int, max_time: float):
    """
    pick file format (same as C):
      trig weight amp
    returns:
      trig: float32 [<=max_n], padded with INF
    """
    trig = np.full((max_n,), INF, np.float32)
    w = np.zeros((max_n,), np.float32)
    amp = np.zeros((max_n,), np.float32)
    n = 0
    if not os.path.exists(path):
        return trig, w, amp, 0
    with open(path, "r") as f:
        for line in f:
            if n >= max_n:
                break
            line = line.strip()
            if not line:
                continue
            a = line.split()
            t = float(a[0])
            if t > max_time:
                continue
            trig[n] = t
            if len(a) > 1:
                w[n] = float(a[1])
            if len(a) > 2:
                amp[n] = float(a[2])
            n += 1
    # ensure sorted
    idx = np.argsort(trig[:n], kind="mergesort")
    trig[:n] = trig[:n][idx]
    w[:n] = w[:n][idx]
    amp[:n] = amp[:n][idx]
    return trig, w, amp, n


def load_all_picks(pick_dir: str, net: list, sta: list, max_n: int, max_time: float):
    """
    Load per-station P/S picks. Output arrays shape (Nst, max_n).
    """
    nst = len(net)
    ptrig = np.full((nst, max_n), INF, np.float32)
    strig = np.full((nst, max_n), INF, np.float32)
    pW = np.zeros((nst, max_n), np.float32)
    sW = np.zeros((nst, max_n), np.float32)
    pA = np.zeros((nst, max_n), np.float32)
    sA = np.zeros((nst, max_n), np.float32)

    for i in range(nst):
        pfile = os.path.join(pick_dir, f"{net[i]}.{sta[i]}.P.txt")
        sfile = os.path.join(pick_dir, f"{net[i]}.{sta[i]}.S.txt")
        t, w, a, _ = read_pick_triplets(pfile, max_n, max_time)
        ptrig[i] = t; pW[i] = w; pA[i] = a
        t, w, a, _ = read_pick_triplets(sfile, max_n, max_time)
        strig[i] = t; sW[i] = w; sA[i] = a

    return ptrig, strig, pW, sW, pA, sA

def load_all_picks_from_singlefile(
    pick_file: str,
    net: list, sta: list,
    max_n: int, max_time: float,
    min_conf: float = 0.0,
):
    p_by_sta, s_by_sta = read_pickfile_grouped(
        pick_file, max_time=max_time, min_conf=min_conf
    )
    return build_pick_arrays_from_grouped(
        net, sta,
        p_by_sta, s_by_sta,
        max_n=max_n, max_time=max_time,
        min_conf=min_conf,
    )

import os
import numpy as np
from datetime import datetime

INF = 1.0e8

def _parse_dt_naive(s: str) -> float:
    # "2021-05-21 09:04:00.165000" -> epoch seconds (naive)
    # 这里用本地时区会引入歧义；我们按 naive datetime 计算“timestamp”不可用。
    # 所以返回“datetime 对象”更稳，但为了数组存储，用 POSIX epoch 需要指定时区。
    # 方案：用 datetime.toordinal + seconds 形成单调数；但输出时仍用原字符串即可。
    # ——最简单：直接保存原字符串；但会占内存且慢。
    #
    # 折中：保存 datetime64[us]（numpy）：
    dt = np.datetime64(s)  # us 精度
    return dt.astype('datetime64[us]').astype(np.int64) / 1e6  # “秒”标尺（非真正 epoch，但可做差）

def load_all_picks_from_singlefile_v2(
    pick_file: str,
    net: list,
    sta: list,
    max_n: int,
    max_time: float,
    min_conf: float = 0.0,
    p_phases=("Pg","P","Pn"),
    s_phases=("Sg","S","Sn"),
):
    """
    读取你这种单文件格式（含 #data/... 头 + 多行 picks）并按台站聚合到矩阵：
      ptrig/strig: float32 秒（用于关联）
      pabs/sabs  : float64 秒标尺（用于输出绝对时间 + dt）
      pconf/sconf: float32 置信度
    """
    nst = len(net)
    key2idx = {f"{net[i]}.{sta[i]}": i for i in range(nst)}

    ptrig = np.full((nst, max_n), INF, np.float32)
    strig = np.full((nst, max_n), INF, np.float32)

    pabs  = np.full((nst, max_n), np.nan, np.float64)
    sabs  = np.full((nst, max_n), np.nan, np.float64)

    pconf = np.zeros((nst, max_n), np.float32)
    sconf = np.zeros((nst, max_n), np.float32)

    pcnt = np.zeros((nst,), np.int32)
    scnt = np.zeros((nst,), np.int32)

    if not os.path.exists(pick_file):
        return ptrig, strig, pconf, sconf, pabs, sabs

    with open(pick_file, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("##"):
                continue
            if line.startswith("#"):
                # 形如：#data/.../YN.YSW03.00....mseed
                continue

            a = line.split(",")
            if len(a) < 9:
                continue

            phase = a[0].strip()
            t_rel = float(a[1])
            conf  = float(a[2])
            t_abs_str = a[3].strip()
            sta_full = a[6].strip()  # "YN.YSW03.00"
            # 你示例里 sta_full 带 ".00"，而 station_file 里通常是 net+sta
            # 这里默认取前两段作为 key：YN.YSW03
            ss = sta_full.split(".")
            if len(ss) >= 2:
                key = f"{ss[0]}.{ss[1]}"
            else:
                key = sta_full

            if t_rel > max_time or conf < min_conf:
                continue
            if key not in key2idx:
                continue
            i = key2idx[key]

            # 绝对时间保存为 numpy datetime64 变换出来的“秒标尺”
            # 仅用于输出 dt 和重构 datetime 字符串（我们会保留原字符串输出）
            tabs = _parse_dt_naive(t_abs_str)

            if phase in p_phases:
                j = int(pcnt[i])
                if j < max_n:
                    ptrig[i, j] = t_rel
                    pconf[i, j] = conf
                    pabs[i, j]  = tabs
                    pcnt[i] += 1
            elif phase in s_phases:
                j = int(scnt[i])
                if j < max_n:
                    strig[i, j] = t_rel
                    sconf[i, j] = conf
                    sabs[i, j]  = tabs
                    scnt[i] += 1

    # 每个台站内部排序（按相对时间排序，同时重排 abs/conf）
    for i in range(nst):
        n = int(pcnt[i])
        if n > 1:
            idx = np.argsort(ptrig[i, :n], kind="mergesort")
            ptrig[i, :n] = ptrig[i, :n][idx]
            pconf[i, :n] = pconf[i, :n][idx]
            pabs[i, :n]  = pabs[i, :n][idx]
        n = int(scnt[i])
        if n > 1:
            idx = np.argsort(strig[i, :n], kind="mergesort")
            strig[i, :n] = strig[i, :n][idx]
            sconf[i, :n] = sconf[i, :n][idx]
            sabs[i, :n]  = sabs[i, :n][idx]

    return ptrig, strig, pconf, sconf, pabs, sabs

def _sec_to_timestr_naive(sec: float) -> str:
    # sec 是我们上面用 datetime64 转的“秒标尺”，可逆为 datetime64 再格式化
    dt64 = np.datetime64(int(sec * 1e6), 'us')
    # 输出成 "YYYY-mm-dd HH:MM:SS.ffffff"
    s = str(dt64).replace('T', ' ')
    if '.' not in s:
        s += '.000000'
    else:
        # numpy datetime64 可能输出到微秒但位数不够
        head, frac = s.split('.')
        s = head + '.' + (frac + '000000')[:6]
    return s

def collect_event_phases(
    real: "FastREAL",
    latref0: float, lonref0: float,
    best_idx: int, ot_rel: float,
    dep0: float,
    sx_km: np.ndarray, sy_km: np.ndarray,
    gx_km: np.ndarray, gy_km: np.ndarray,
    # picks + abs/conf
    ptrig0: np.ndarray, strig0: np.ndarray,
    pabs: np.ndarray, sabs: np.ndarray,
    pconf: np.ndarray, sconf: np.ndarray,
):
    """
    返回该事件的相列表，每项：
      (phase, pick_time_str, dt, dist_km, conf, resid, sta_id)
    """
    out = []

    xg = float(gx_km[best_idx])
    yg = float(gy_km[best_idx])

    for st in range(real.Nst):
        dx = xg - float(sx_km[st])
        dy = yg - float(sy_km[st])

        # 3D 距离（km）
        dist_km = (dx*dx + dy*dy + dep0*dep0) ** 0.5

        # 理论走时
        tp_cal = dist_km / real.vp0 + float(real.elev_km[st]) / real.s_vp0
        ts_cal = dist_km / real.vs0 + float(real.elev_km[st]) / real.s_vs0

        # 预测到时（相对秒）
        tp_pre = ot_rel + tp_cal
        ts_pre = ot_rel + ts_cal

        # 窗口
        tp_b = tp_pre - real.nrt * real.ptw * 0.5
        tp_e = tp_pre + real.nrt * real.ptw * 0.5
        ts_b = ts_pre - real.nrt * real.stw * 0.5
        ts_e = ts_pre + real.nrt * real.stw * 0.5

        # --- P: 取窗口内第一条 ---
        row = ptrig0[st]
        j = int(np.searchsorted(row, tp_b, side="left"))
        if j < row.shape[0]:
            tpk = float(row[j])
            if (tpk < tp_e) and (tpk < INF):
                # dt = pick - origin
                dt = tpk - ot_rel
                # 残差：obs - cal = (pick-ot) - tcal
                resid = dt - tp_cal

                # 绝对时间字符串：优先用 pabs，如果缺失就空
                if not np.isnan(pabs[st, j]):
                    pick_str = _sec_to_timestr_naive(float(pabs[st, j]))
                else:
                    pick_str = ""

                conf = float(pconf[st, j])
                sta_id = f"{real.sta_net[st]}.{real.sta_name[st]}"  # 需要你在 FastREAL 里存一下
                out.append(("Pg", pick_str, dt, dist_km, conf, resid, sta_id))

        # --- S: 同理，且满足 dtps 约束 ---
        row = strig0[st]
        j = int(np.searchsorted(row, ts_b, side="left"))
        if j < row.shape[0]:
            tsk = float(row[j])
            if (tsk < ts_e) and (tsk < INF):
                # dtps 约束（跟你内核一致：预测差值>dtps 且与P不太近）
                if (ts_pre - tp_pre) > real.dtps:
                    dt = tsk - ot_rel
                    resid = dt - ts_cal
                    if not np.isnan(sabs[st, j]):
                        pick_str = _sec_to_timestr_naive(float(sabs[st, j]))
                    else:
                        pick_str = ""
                    conf = float(sconf[st, j])
                    sta_id = f"{real.sta_net[st]}.{real.sta_name[st]}"
                    out.append(("Sg", pick_str, dt, dist_km, conf, resid, sta_id))

    return out
# ----------------------------
# Heap scheduler for initiating P
# ----------------------------
class PickScheduler:
    def __init__(self, ptrig0: np.ndarray):
        self.ptrig0 = np.asarray(ptrig0, np.float32)
        self.Nst, self.NNps = self.ptrig0.shape
        self.prem = np.zeros((self.Nst, self.NNps), np.bool_)
        self.pcur = np.zeros((self.Nst,), np.int32)
        self.heap = []
        self._init_heap()

    def _advance(self, st: int) -> bool:
        j = int(self.pcur[st])
        row = self.ptrig0[st]
        rem = self.prem[st]
        while j < self.NNps:
            t = float(row[j])
            if t >= INF:
                break
            if not rem[j]:
                self.pcur[st] = j
                return True
            j += 1
        self.pcur[st] = self.NNps
        return False

    def _init_heap(self):
        self.heap.clear()
        for st in range(self.Nst):
            self.pcur[st] = 0
            if self._advance(st):
                j = int(self.pcur[st])
                heapq.heappush(self.heap, (float(self.ptrig0[st, j]), st, j))

    def pop(self):
        while self.heap:
            t, st, idx = heapq.heappop(self.heap)
            if idx != int(self.pcur[st]):
                continue
            if self.prem[st, idx]:
                continue
            if t >= INF:
                return None
            return t, st, idx
        return None

    def remove(self, st: int, idx: int):
        if idx < 0 or idx >= self.NNps:
            return
        if self.prem[st, idx]:
            return
        self.prem[st, idx] = True
        if idx == int(self.pcur[st]):
            if self._advance(st):
                j = int(self.pcur[st])
                heapq.heappush(self.heap, (float(self.ptrig0[st, j]), st, j))

    def remove_first_in_window(self, st: int, tb: float, te: float):
        row = self.ptrig0[st]
        j = int(np.searchsorted(row, tb, side="left"))
        while j < self.NNps:
            t = float(row[j])
            if t >= te or t >= INF:
                return False
            if not self.prem[st, j]:
                self.remove(st, j)
                return True
            j += 1
        return False


# ----------------------------
# Numba kernels
# ----------------------------
@njit(inline="always")
def lower_bound(a, x):
    l = 0
    r = a.shape[0]
    while l < r:
        m = (l + r) >> 1
        if a[m] < x:
            l = m + 1
        else:
            r = m
    return l

@njit(inline="always")
def upper_bound(a, x):
    l = 0
    r = a.shape[0]
    while l < r:
        m = (l + r) >> 1
        if a[m] <= x:
            l = m + 1
        else:
            r = m
    return l

@njit(inline="always")
def travel_time_homo(dx_km, dy_km, dep_km, v, elev_km, sv):
    # rdist = sqrt(dx^2 + dy^2 + dep^2)
    r = np.sqrt(dx_km*dx_km + dy_km*dy_km + dep_km*dep_km)
    return r / v + elev_km / sv

@njit(parallel=True, cache=True)
def eval_grid_homo_fast(
    # grid (local km coordinates)
    gx_km, gy_km, gdep_km,  # shape (nnn,)
    # stations (local km)
    sx_km, sy_km, selev_km, # shape (Nst,)
    # picks (sorted, padded with INF)
    ptrig0, strig0,         # shape (Nst, NNps)
    np_start, np_end, ns_start, ns_end,  # shape (Nst,)
    # initiating pick info
    tpmin0, ref_st,
    # model params
    vp0, vs0, s_vp0, s_vs0,
    nrt, ptw, stw,
    dtps, gcarc0_km,
    # scoring
    np0_th, ns0_th, nps0_th, npsboth0_th,
    use_strict_median: int = 0
):
    """
    Return best grid index and best score fields:
      best_idx, best_otime, best_std, best_ps, best_pcount, best_scount, best_psboth
    """
    nnn = gx_km.shape[0]
    nst = sx_km.shape[0]
    nnps = ptrig0.shape[1]

    best_idx = -1
    best_ps = -1
    best_std = 1.0e9
    best_ot = -1.0e9
    best_p = 0
    best_s = 0
    best_psboth = 0

    # reference station travel time to each grid point (needed for tp_pre formulation)
    # in C: tp_pre = tpmin0 - tp0_cal + tp_cal
    # tp0_cal depends on grid point and reference station.
    # Here we compute per grid point inside loop.

    for l in prange(nnn):
        # per gridpoint accumulators (fixed-size buffers)
        # NOTE: to keep Numba fast, we use preallocated small arrays using np.empty,
        # but size depends on Nst. For very large Nst, use two-pass approach.
        torg = np.empty((2*nst,), np.float32)
        used_phase = np.zeros((nst,), np.uint8)  # bit0: used P, bit1: used S
        baz_dummy = np.empty((2*nst,), np.float32)  # placeholder for gap; we skip gap for speed
        ps = 0
        pcount = 0
        scount = 0
        psboth = 0

        # grid point
        xg = gx_km[l]
        yg = gy_km[l]
        dep = gdep_km[l]

        # reference station traveltime to grid point
        dx0 = xg - sx_km[ref_st]
        dy0 = yg - sy_km[ref_st]
        tp0 = travel_time_homo(dx0, dy0, dep, vp0, selev_km[ref_st], s_vp0)

        # scan stations
        for i in range(nst):
            dx = xg - sx_km[i]
            dy = yg - sy_km[i]
            # distance filter (like GCarc0): use horizontal km only for speed
            # rdist_h = sqrt(dx^2+dy^2)
            rdh = np.sqrt(dx*dx + dy*dy)
            if rdh > gcarc0_km:
                continue

            tp_cal = travel_time_homo(dx, dy, dep, vp0, selev_km[i], s_vp0)
            ts_cal = travel_time_homo(dx, dy, dep, vs0, selev_km[i], s_vs0)

            tp_pre = tpmin0 - tp0 + tp_cal
            ts_pre = tpmin0 - tp0 + ts_cal

            tp_b = tp_pre - nrt * ptw * 0.5
            tp_e = tp_pre + nrt * ptw * 0.5
            ts_b = ts_pre - nrt * stw * 0.5
            ts_e = ts_pre + nrt * stw * 0.5

            # find one P pick in window
            ptemp = -1.0
            puse = 0
            j0 = np_start[i]
            j1 = np_end[i]
            if j1 > j0:
                # binary search inside [j0, j1)
                # find first >= tp_b
                arr = ptrig0[i]
                j = j0 + lower_bound(arr[j0:j1], tp_b)
                if j < j1:
                    tpk = arr[j]
                    if tpk < tp_e and tpk < INF:
                        torg[ps] = tpk - tp_cal
                        ps += 1
                        pcount += 1
                        puse = 1
                        ptemp = tpk
                        used_phase[i] |= 1

            # find one S pick in window, with dtps constraints
            j0s = ns_start[i]
            j1s = ns_end[i]
            if j1s > j0s:
                if (ts_pre - tp_pre) > dtps:
                    arrs = strig0[i]
                    j = j0s + lower_bound(arrs[j0s:j1s], ts_b)
                    if j < j1s:
                        tsk = arrs[j]
                        if tsk < ts_e and tsk < INF:
                            if ptemp < 0 or np.abs(ptemp - tsk) > dtps:
                                torg[ps] = tsk - ts_cal
                                ps += 1
                                scount += 1
                                used_phase[i] |= 2
                                if puse == 1:
                                    psboth += 1

        # thresholds
        if pcount < np0_th or scount < ns0_th or ps < nps0_th or psboth < npsboth0_th:
            continue

        # scoring: origin time estimate + scatter
        # FAST: trimmed mean + std (no sorting full)
        # Strict median option is expensive; keep off unless needed.
        ot = 0.0
        if use_strict_median == 0:
            # trimmed mean: drop top/bottom 10% by partial selection is complex;
            # use mean first-pass then clip residuals (Huber-like)
            mu = 0.0
            for k in range(ps):
                mu += torg[k]
            mu /= ps

            # robustify: compute scale, clip at 2.5*scale
            var = 0.0
            for k in range(ps):
                d = torg[k] - mu
                var += d*d
            var /= max(ps-1, 1)
            s = np.sqrt(var) + 1e-6
            c = 2.5 * s

            mu2 = 0.0
            wsum = 0.0
            for k in range(ps):
                d = torg[k] - mu
                if d > c:
                    x = mu + c
                elif d < -c:
                    x = mu - c
                else:
                    x = torg[k]
                mu2 += x
                wsum += 1.0
            ot = mu2 / wsum

            var2 = 0.0
            for k in range(ps):
                d = torg[k] - ot
                var2 += d*d
            std = np.sqrt(var2 / max(ps-1, 1))
        else:
            # strict median via sort (slow)
            tmp = np.empty((ps,), np.float32)
            for k in range(ps):
                tmp[k] = torg[k]
            tmp.sort()
            if ps & 1:
                ot = tmp[ps//2]
            else:
                ot = 0.5*(tmp[ps//2-1] + tmp[ps//2])
            var2 = 0.0
            for k in range(ps):
                d = torg[k] - ot
                var2 += d*d
            std = np.sqrt(var2 / max(ps-1, 1))

        # choose best: maximize ps, then minimize std
        if (ps > best_ps) or (ps == best_ps and std < best_std):
            best_ps = ps
            best_std = std
            best_ot = ot
            best_idx = l
            best_p = pcount
            best_s = scount
            best_psboth = psboth

    return best_idx, best_ot, best_std, best_ps, best_p, best_s, best_psboth


# ----------------------------
# Main REAL-like runner (Python)
# ----------------------------
class FastREAL:
    def __init__(
        self,
        stla, stlo, elev_m,
        ptrig0, strig0,
        pabs, sabs, pconf, sconf,
        # control params
        lat_center_deg: float,
        rx_deg: float, rh_km: float,
        dx_deg: float, dh_km: float,
        tint_sec: float,
        vp0: float, vs0: float,
        s_vp0: float, s_vs0: float,
        np0: int, ns0: int, nps0: int, npsboth0: int,
        std0: float, dtps: float, nrt: float,
        rsel: float = 5.0,
        gcarc0_deg: float = 180.0,
        ispeed: bool = True,
        max_time: float = 2700000.0,
        use_strict_median: bool = False,
        net=None, 
        sta=None, 
    ):
        self.pabs = pabs 
        self.sabs = sabs 
        self.pconf = pconf 
        self.sconf = sconf 
        self.stla = np.asarray(stla, np.float32)
        self.stlo = np.asarray(stlo, np.float32)
        self.elev_km = np.asarray(elev_m, np.float32) / 1000.0

        self.ptrig0 = np.asarray(ptrig0, np.float32)
        self.strig0 = np.asarray(strig0, np.float32)
        self.Nst, self.NNps = self.ptrig0.shape

        self.lat_center = float(lat_center_deg)
        self.rx1 = float(rx_deg)
        self.rh = float(rh_km)
        self.dx1 = float(dx_deg)
        self.dh = float(dh_km)
        self.tint = float(tint_sec)

        self.vp0 = float(vp0)
        self.vs0 = float(vs0)
        self.s_vp0 = float(s_vp0)
        self.s_vs0 = float(s_vs0)

        self.np0 = int(np0); self.ns0 = int(ns0); self.nps0 = int(nps0); self.npsboth0 = int(npsboth0)
        self.std0 = float(std0)
        self.dtps = float(dtps)
        self.nrt = float(nrt)
        self.rsel = float(rsel)

        self.gcarc0_km = float(gcarc0_deg) * DEG2KM
        self.ispeed = bool(ispeed)
        self.max_time = float(max_time)
        self.use_strict_median = bool(use_strict_median)

        # derived
        self.dx2 = self.dx1 / np.cos(np.deg2rad(self.lat_center))
        self.rx2 = self.rx1 / np.cos(np.deg2rad(self.lat_center))

        # grid size
        self.nlat = int(2*self.rx1/self.dx1 + 1)
        self.nlon = int(2*self.rx2/self.dx2 + 1)
        self.ndep = int(self.rh/self.dh + 1)
        self.nnn = self.nlat * self.nlon * self.ndep

        # windows (like C)
        self.ptw = np.sqrt((self.dx1*DEG2KM)**2 + (self.dx1*DEG2KM)**2 + self.dh**2) / self.vp0
        self.stw = np.sqrt((self.dx1*DEG2KM)**2 + (self.dx1*DEG2KM)**2 + self.dh**2) / self.vs0
        if self.tint < self.stw:
            self.tint = self.stw
        self.sta_net  = list(net) if net is not None else [f"ST{idx}" for idx in range(self.Nst)]
        self.sta_name = list(sta) if sta is not None else [f"ST{idx}" for idx in range(self.Nst)]


    def _build_local_km_coords(self, latref0, lonref0):
        """
        local tangent-plane approx:
          x_km = (lon - lonref0) * cos(lat_center) * DEG2KM
          y_km = (lat - latref0) * DEG2KM
        Use fixed lat_center for speed (matches your C idea).
        """
        cosc = np.cos(np.deg2rad(self.lat_center)).astype(np.float64)
        sx = (self.stlo.astype(np.float64) - lonref0) * cosc * DEG2KM
        sy = (self.stla.astype(np.float64) - latref0) * DEG2KM
        return sx.astype(np.float32), sy.astype(np.float32)

    def _build_grid_local(self, latref0, lonref0):
        """
        Grid in local km coordinates relative to (latref0, lonref0).
        """
        cosc = np.cos(np.deg2rad(self.lat_center)).astype(np.float64)

        lat_vals = (latref0 - self.rx1) + np.arange(self.nlat, dtype=np.float64) * self.dx1
        lon_vals = (lonref0 - self.rx2) + np.arange(self.nlon, dtype=np.float64) * self.dx2
        dep_vals = np.arange(self.ndep, dtype=np.float64) * self.dh

        # convert to local km coordinates (relative to ref0)
        gy = (lat_vals - latref0) * DEG2KM
        gx = (lon_vals - lonref0) * cosc * DEG2KM
        gdep = dep_vals

        # mesh to flat arrays
        gx3 = np.empty((self.nnn,), np.float32)
        gy3 = np.empty((self.nnn,), np.float32)
        gd3 = np.empty((self.nnn,), np.float32)
        idx = 0
        for i in range(self.nlat):
            for j in range(self.nlon):
                for k in range(self.ndep):
                    gx3[idx] = gx[j]
                    gy3[idx] = gy[i]
                    gd3[idx] = gdep[k]
                    idx += 1
        return gx3, gy3, gd3

    def run(self, latref0_init: float = None, lonref0_init: float = None, max_events: int = 10_000):
        """
        Return events list:
          (otime_sec, lat, lon, dep_km, std, pcount, scount, pscount, psboth)
        """
        sched = PickScheduler(self.ptrig0)
        events = []
        per_event_phase_rows = []
        # if no global ref given: use initiating station as ref0 each time (like your inoref logic)
        have_global_ref = (latref0_init is not None) and (lonref0_init is not None)

        # pre-alloc arrays for search ranges
        np_start = np.zeros((self.Nst,), np.int32)
        np_end   = np.zeros((self.Nst,), np.int32)
        ns_start = np.zeros((self.Nst,), np.int32)
        ns_end   = np.zeros((self.Nst,), np.int32)

        while True:
            got = sched.pop()
            if got is None:
                break
            tpmin0, m, n = got
            if tpmin0 >= self.max_time:
                break

            # reference
            if have_global_ref:
                latref0 = float(latref0_init)
                lonref0 = float(lonref0_init)
            else:
                latref0 = float(self.stla[m])
                lonref0 = float(self.stlo[m])

            # time window bounds (same idea as C; distmax 可更精确，这里用一个保守上界)
            # 你可以把 distmax 换成 station box 的最大距离估计。
            # 这里保守用 rx/rh 估计水平最大距离：
            dist_h_km = np.sqrt((self.rx1*DEG2KM)**2 + (self.rx2*DEG2KM)**2)
            tpmin = max(0.0, tpmin0 - 1.2 * dist_h_km / self.vp0)
            tpmax = min(self.max_time, tpmin0 + 1.2 * dist_h_km / self.vp0)
            tsmin = max(0.0, tpmin0 - 1.2 * dist_h_km / self.vs0)
            tsmax = min(self.max_time, tpmin0 + 1.2 * dist_h_km / self.vs0)

            # per-station [start,end) by searchsorted (fast)
            for i in range(self.Nst):
                p = self.ptrig0[i]
                s = self.strig0[i]
                np_start[i] = int(np.searchsorted(p, tpmin, side="left"))
                np_end[i]   = int(np.searchsorted(p, tpmax, side="right"))
                ns_start[i] = int(np.searchsorted(s, tsmin, side="left"))
                ns_end[i]   = int(np.searchsorted(s, tsmax, side="right"))

            # local coords
            sx_km, sy_km = self._build_local_km_coords(latref0, lonref0)
            gx_km, gy_km, gd_km = self._build_grid_local(latref0, lonref0)

            # grid eval (Numba)
            best_idx, ot, std, psc, pc, sc, psboth = eval_grid_homo_fast(
                gx_km, gy_km, gd_km,
                sx_km, sy_km, self.elev_km,
                self.ptrig0, self.strig0,
                np_start, np_end, ns_start, ns_end,
                float(tpmin0), int(m),
                self.vp0, self.vs0, self.s_vp0, self.s_vs0,
                self.nrt, self.ptw, self.stw,
                self.dtps, self.gcarc0_km,
                self.np0, self.ns0, self.nps0, self.npsboth0,
                1 if self.use_strict_median else 0
            )

            # accept/reject (match C thresholds)
            accepted = (best_idx >= 0 and
                        pc >= self.np0 and sc >= self.ns0 and psc >= self.nps0 and
                        psboth >= self.npsboth0 and std <= self.std0)

            if accepted:
                # map best_idx -> lat/lon/dep
                i = best_idx // (self.nlon * self.ndep)
                j = (best_idx - i * self.nlon * self.ndep) // self.ndep
                k = best_idx - i * self.nlon * self.ndep - j * self.ndep

                lat0 = (latref0 - self.rx1) + i * self.dx1
                lon0 = (lonref0 - self.rx2) + j * self.dx2
                dep0 = k * self.dh

                events.append((ot, lat0, lon0, dep0, std, int(pc), int(sc), int(psc), int(psboth)))

                # 抽取该事件关联到的 Pg/Sg（用于输出相行）
                ph_rows = collect_event_phases(
                    self, latref0, lonref0,
                    best_idx, ot, dep0,
                    sx_km, sy_km,
                    gx_km, gy_km,
                    self.ptrig0, self.strig0,
                    self.pabs, self.sabs,
                    self.pconf, self.sconf
                )
                per_event_phase_rows.append(ph_rows)

                # ispeed: remove one P pick in predicted window for each station
                if self.ispeed:
                    xg = float(gx_km[best_idx])
                    yg = float(gy_km[best_idx])
                    for st in range(self.Nst):
                        dx = xg - float(sx_km[st])
                        dy = yg - float(sy_km[st])
                        tp_cal = (dx*dx + dy*dy + dep0*dep0) ** 0.5 / self.vp0 + float(self.elev_km[st]) / self.s_vp0
                        tp_pre = ot + tp_cal
                        tb = max(0.0, tp_pre - self.nrt*self.ptw*0.5)
                        te = min(self.max_time, tp_pre + self.nrt*self.ptw*0.5)
                        sched.remove_first_in_window(st, tb, te)


            # always remove initiating P itself
            sched.remove(m, n)

            if len(events) >= max_events:
                break

        return events, per_event_phase_rows


def write_events_real_format(
    out_path: str,
    events: list,
    real: "FastREAL",
    # picks abs/conf
    pabs: np.ndarray, sabs: np.ndarray,
    pconf: np.ndarray, sconf: np.ndarray,
    # 你在 run() 里为了 post-pass 需要拿到的网格与局部坐标
    # ——最省事：在 real.run() 接受事件时，把这些也保存/返回（见下方建议）
    per_event_phase_rows: list,
):
    """
    per_event_phase_rows: 与 events 同长度的列表，
      per_event_phase_rows[eid] = [(phase, pick_str, dt, dist, conf, resid, sta_id), ...]
    """
    with open(out_path, "w") as f:
        for eid, ev in enumerate(events, start=1):
            ot_rel, lat0, lon0, dep0, std, pc, sc, psc, psboth = ev

            # 事件绝对时间：用该事件的第一条相（如果有）反推 origin_abs
            # origin_abs = pick_abs - dt
            ev_ph = per_event_phase_rows[eid-1]
            if ev_ph:
                # 找一个有 pick_str 的
                pick_str, dt = None, None
                for row in ev_ph:
                    if row[1]:
                        pick_str = row[1]; dt = row[2]
                        break
                if pick_str is not None:
                    # pick_str -> datetime64 秒标尺
                    pick_sec = _parse_dt_naive(pick_str)
                    origin_sec = pick_sec - float(dt)
                    origin_str = _sec_to_timestr_naive(origin_sec)
                else:
                    origin_str = ""
            else:
                origin_str = ""

            # 你示例头：#EVENT,123456,2021-05-22 ..., lon,lat,dep,6,10,16,5
            # 这里第二列用 eid（也可换成 hash/时间戳）
            f.write(
                f"#EVENT,{eid},{origin_str},{lon0:.3f},{lat0:.3f},{dep0:.3f},"
                f"{pc},{sc},{psc},{psboth}\n"
            )

            for (ph, pick_time_str, dt, dist_km, conf, resid, sta_id) in ev_ph:
                f.write(
                    f"{ph},{pick_time_str},{dt:.3f},{dist_km:.3f},{conf:.3f},{resid:.3f},{sta_id}\n"
                )

# ----------------------------
# Example usage
# ----------------------------
def main_example_orignal():
    # --- user settings (match your C usage) ---
    MAXTIME = 2700000.0
    station_file = "data/select.loc"
    pick_dir = "realdata/20210522"

    stla, stlo, elev, net, sta = read_station_txt(station_file)

    # decide NNps (max picks per station); you can keep 20000 like C
    NNps = 20000
    ptrig0, strig0, pW, sW, pA, sA = load_all_picks(pick_dir, net, sta, NNps, MAXTIME)

    real = FastREAL(
        stla, stlo, elev,
        ptrig0, strig0,
        lat_center_deg=0.0,
        rx_deg=1.0, rh_km=30.0,
        dx_deg=0.05, dh_km=2.0,
        tint_sec=10.0,
        vp0=6.0, vs0=3.5,
        s_vp0=6.0, s_vs0=3.5,
        np0=6, ns0=4, nps0=10, npsboth0=2,
        std0=1.0, dtps=2.0, nrt=2.0,
        gcarc0_deg=3.0,
        ispeed=True,
        max_time=MAXTIME,
        use_strict_median=False,
    )

    events = real.run(latref0_init=None, lonref0_init=None, max_events=100000)
    print("events:", len(events))
    for e in events[:10]:
        print(e)
import time
from datetime import datetime

def _ts():
    # 精确到毫秒
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

class StepTimer:
    def __init__(self, title="main_example"):
        self.title = title
        self.t0 = time.perf_counter()
        self.last = self.t0
        print(f"[{_ts()}] [{self.title}] START")

    def step(self, name: str):
        now = time.perf_counter()
        dt = now - self.last
        tot = now - self.t0
        print(f"[{_ts()}] [{self.title}] STEP done: {name:>28s} | step={dt:8.3f}s | total={tot:8.3f}s")
        self.last = now

    def mark(self, name: str):
        # 只打印一个“到达某点”的时间，不计算 step（有时你想分开 begin/end）
        now = time.perf_counter()
        tot = now - self.t0
        print(f"[{_ts()}] [{self.title}] MARK: {name} | total={tot:8.3f}s")

    def end(self):
        now = time.perf_counter()
        tot = now - self.t0
        print(f"[{_ts()}] [{self.title}] END | total={tot:8.3f}s")


def main_example():
    tm = StepTimer("REAL-Python")

    MAXTIME = 2700000.0
    station_file = "data/select.loc"
    pick_file = "odata/pnsn.v3.txt"  # 你的这种文件
    tm.step("set paths & constants")

    # 1) stations
    tm.mark("read_station_txt begin")
    stla, stlo, elev, net, sta = read_station_txt(station_file)
    tm.step(f"read_station_txt (Nst={len(net)})")

    # 2) picks
    NNps = 20000
    tm.mark("load_all_picks_from_singlefile begin")
    ptrig0, strig0, pconf, sconf, pabs, sabs = load_all_picks_from_singlefile_v2(
        pick_file, net, sta, NNps, MAXTIME, min_conf=0.0
    )
    tm.step(f"load_all_picks_from_singlefile (NNps={NNps})")

    # 3) init REAL
    tm.mark("FastREAL init begin")
    real = FastREAL(
        stla, stlo, elev,
        ptrig0, strig0,
        pabs, sabs, pconf, sconf, 
        lat_center_deg=0.0,
        rx_deg=0.5, rh_km=25.0,
        dx_deg=0.05, dh_km=5.0,
        tint_sec=10.0,
        vp0=6.0, vs0=3.5,
        s_vp0=6.0, s_vs0=3.5,
        np0=4, ns0=4, nps0=8, npsboth0=2,
        std0=1.0, dtps=2.0, nrt=2.0,
        gcarc0_deg=3.0,
        ispeed=True,
        max_time=MAXTIME,
        use_strict_median=False,
        net=net, sta=sta,
    )
    tm.step("FastREAL init")

    # 4) run
    tm.mark("real.run begin")
    events, per_event_phase_rows = real.run(latref0_init=None, lonref0_init=None, max_events=100000)
    write_events_real_format(
        out_path="odata/catalog_phase_like_real.txt",
        events=events,
        real=real,
        pabs=pabs, sabs=sabs,
        pconf=pconf, sconf=sconf,
        per_event_phase_rows=per_event_phase_rows
    )
    tm.step(f"real.run (events={len(events)})")

    print(f"[{_ts()}] events: {len(events)}")
    tm.end()


if __name__ == "__main__":
    main_example()
