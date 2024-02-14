# Attention is All You Align

Project repository for CSE 599c taught in the 2024 winter. 
Root folder contains:
- `aligner`: to be treated as a single module (imported or called by scripts _outside_ this folder). Contains all the key implementations
- `data`: Upload data here. Do not explicitly upload data if the data is large; use `.gitignore` instead (e.g., for audio files). Write an easy-to-run data downloader script that everybody can use.

## Coding Conventions
- Readable code. The use of `typing` (type hinting) and detailed comments is encouraged.
- Common Conda environment for `Python 3.10`. Expected to run on `attu`/`hyak`.

## Google Drive
- [Click](https://drive.google.com/drive/folders/1pgjzIMsOfZdnw3tQTR1H1LK3UXPCl6SL?usp=sharing) to view.

## Training Script
```
python train.py --save_to_path ../ckpt/v0/0 --save_every_epoch 1 -- batch_size 8
```
