import os
logs_base_dir = "logs"
# 로그 기록을 위한 directory 생성
os.makedirs(logs_base_dir, exist_ok = True)

from torch.utils.tensorboard import SummaryWriter # 기록 생성 객체 SummaryWriter 생성
import numpy as np

# metric 그리기
exp  = f"{logs_base_dir}/ex1"
writer = SummaryWriter(exp)
# n_iter : epoch (x축)
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
writer.flush()

# 여러 그래프를 한번에 그리는 방법
from torch.utils.tensorboard import SummaryWriter
exp = f"{logs_base_dir}/ex2"
writer1 = SummaryWriter(exp)
r = 5
for i in range(100):
    writer1.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)
writer1.close()

# 하이퍼파리미터 기록하기
from torch.utils.tensorboard import SummaryWriter
with SummaryWriter(logs_base_dir) as w:
    for i in range(5):
        w.add_hparams({'lr': 0.1*i, 'bsize': i},
                      {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

# 이미지 그리기
img_batch = np.zeros((16, 3, 100, 100))
for i in range(16):
    img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
    img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

writer = SummaryWriter(logs_base_dir)
writer.add_images('my_image_batch', img_batch, 0)
writer.close()

# 3D mesh 이미지 그리기
import torch
vertices_tensor = torch.as_tensor([
    [1, 1, 1],
    [-1, -1, 1],
    [1, -1, -1],
    [-1, 1, -1],
], dtype=torch.float).unsqueeze(0)
colors_tensor = torch.as_tensor([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 255],
], dtype=torch.int).unsqueeze(0)
faces_tensor = torch.as_tensor([
    [0, 2, 3],
    [0, 3, 1],
    [0, 1, 2],
    [1, 3, 2],
], dtype=torch.int).unsqueeze(0)

writer = SummaryWriter(logs_base_dir)
writer.add_mesh('my_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)

writer.close()