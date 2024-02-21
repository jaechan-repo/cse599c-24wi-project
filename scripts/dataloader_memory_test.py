import sys
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
from aligner.dataset import MaestroDataset
from memory_profiler import profile

dataset = MaestroDataset(root_dir="../data/maestro-v3.0.0",
                         split='train')
dataloader = DataLoader(dataset,
                        batch_size=4,
                        collate_fn=MaestroDataset.collate_fn)
it = iter(dataloader)

for i in range(4):
    batch = next(it)
    print(batch.Y.shape)
