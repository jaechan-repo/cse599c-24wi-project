import sys
sys.path.append("..")

from aligner.data_loader.data_loader import MaestroDataset

maestro = MaestroDataset("/Users/alan/Workspace/scorealign/data")

it = iter(maestro)
print(len(next(it)))