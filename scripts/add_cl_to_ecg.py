#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_cl_to_ecg.py — 给已生成的 <PATIENT>-ECG.pickle 补上逐点 CL 列(以及 3 个 beat 标注),
无需重读全量 export、无需重建 pickle。

来源: 每个点的 <ReferenceAnnotations Beat0/Beat-1/Beat-2 CycleLength/> (Point_Export.xml)。
做法: 按 DataFrame 里的 point id,从 export 压缩包读对应 Point_Export.xml 的 CL,新增列后回写。

新增列: cycle_length (int, ms; -1=该点无标注), beat0/beat_1/beat_2 (int)

用法:
  add_cl_to_ecg.py <PATIENT-ECG.pickle> <export.zip> <MAP_NAME>
例:
  add_cl_to_ecg.py PATIENT-ECG.pickle "/path/to/export.zip" 4-LA-BIPOLAR
"""
import sys, os, re, pickle, zipfile, tempfile
import numpy as np

def is7z(p): return open(p, "rb").read(2) == b"7z"

def build_cl_map(zip_path, map_name):
    """返回 {point_id: (cycle_length, beat0, beat_1, beat_2)},从 export 里所有该图的 Point_Export.xml。"""
    pat = re.compile(rf"{re.escape(map_name)}_P(\d+)_Point_Export\.xml$")
    out = {}
    def parse(pid, text):
        m = re.search(r"<ReferenceAnnotations ([^/>]*)/>", text)
        if not m:
            out[pid] = (-1, -1, -1, -1); return
        a = m.group(1)
        def g(k):
            mm = re.search(rf'{k}="(-?\d+)"', a); return int(mm.group(1)) if mm else -1
        out[pid] = (g("CycleLength"), g("Beat0"), g("Beat-1"), g("Beat-2"))
    if is7z(zip_path):
        import py7zr
        with py7zr.SevenZipFile(zip_path, "r") as z:
            targets = [n for n in z.getnames() if pat.search(n)]
            with tempfile.TemporaryDirectory() as td:
                z.extract(path=td, targets=targets)
                for t in targets:
                    pid = int(pat.search(t).group(1))
                    fp = os.path.join(td, t)
                    if os.path.exists(fp): parse(pid, open(fp, encoding="latin1").read())
    else:
        z = zipfile.ZipFile(zip_path)
        for n in z.namelist():
            m = pat.search(n)
            if m: parse(int(m.group(1)), z.read(n).decode("latin1", "replace"))
    return out

def main(pkl, zip_path, map_name):
    print(f"1) 从 {os.path.basename(zip_path)} 读 {map_name} 各点 CL ...", flush=True)
    clmap = build_cl_map(zip_path, map_name)
    print(f"   读到 {len(clmap)} 个点的 ReferenceAnnotations", flush=True)

    print(f"2) 载入 {os.path.basename(pkl)} ...", flush=True)
    df = pickle.load(open(pkl, "rb"))
    ids = df["id"].to_numpy()
    cl = np.array([clmap.get(int(i), (-1, -1, -1, -1)) for i in ids])
    df["cycle_length"] = cl[:, 0]
    df["beat0"] = cl[:, 1]; df["beat_1"] = cl[:, 2]; df["beat_2"] = cl[:, 3]
    miss = int((cl[:, 0] == -1).sum())
    valid = cl[cl[:, 0] > 0, 0]
    print(f"   匹配上 {len(df)-miss}/{len(df)} 点 (未匹配 {miss}); "
          f"CL 中位 {int(np.median(valid))} ms, 范围 {valid.min()}–{valid.max()}", flush=True)

    print(f"3) 回写 {pkl} ...", flush=True)
    with open(pkl, "wb") as h:
        pickle.dump(df, h, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"完成: 新列 cycle_length/beat0/beat_1/beat_2;  shape={df.shape}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
