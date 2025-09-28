#!/usr/bin/env python
# bin2st.py
import sys, pathlib, torch
from safetensors.torch import save_file

if len(sys.argv) != 2:
    print("用法: python bin2st.py  <xxx.bin>")
    sys.exit(1)

bin_path = pathlib.Path(sys.argv[1])
st_path  = bin_path.with_suffix('.safetensors')

print(f"加载 {bin_path} ...")
state_dict = torch.load(bin_path, map_location='cpu')

print(f"写入 {st_path} ...")
save_file(state_dict, st_path)

print("✔ 转换完成，可直接重命名使用。")
