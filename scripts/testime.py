import sys
sys.path.append("..")
from aligner.dataset import MaestroDataset
from more_itertools import peekable
from memory_profiler import profile
from torch.utils.data import DataLoader
from aligner.utils.seed import set_seed

set_seed(42)
dataset = MaestroDataset(root_dir="/mmfs1/gscratch/ark/chan0369/projects/cse599c-24wi-project/data/maestro-v3.0.0",
                         split='train')
dataloader = DataLoader(dataset,
                  batch_size=4,
                  num_workers=0,
                  collate_fn=MaestroDataset.collate_fn)

it = peekable(dataloader)

# for i in range(25):
#     batch = next(it)

@profile()
def peeker():
    return it.peek()

res = peeker()