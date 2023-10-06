#! /usr/bin/env python3

import time
import torch
import argparse

from pathlib import Path

from yogo.model import YOGO
from yogo.utils import choose_device


torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument("pth_path", help="path to pth", type=Path)
parser.add_argument("--N", help="num inferences", type=int, default=100)
parser.add_argument("--BS", help="batch size", type=int, default=1)
parser.add_argument("--print-header", action="store_true")
args = parser.parse_args()

device = choose_device()

model, cfg = YOGO.from_pth(Path(args.pth_path), inference=True)
model.resize_model(193)
model.eval()
model = torch.compile(model)
model.to(device)

BS = args.BS
N = args.N

inp = torch.rand(BS, 1, 193, 1032).to(device)

# warmup
for _ in range(100):
    model(inp)

t0 = time.perf_counter()

for _ in range(N):
    model(inp)

t1 = time.perf_counter()

if args.print_header:
    print("num_inferences,batch_size,tot_images,tot_time,FPS")
print(f"{N},{BS},{BS * N},{t1 - t0},{BS * N / (t1 - t0)}")
exit()
