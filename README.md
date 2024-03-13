# [Attention is All You Align](https://drive.google.com/file/d/1g_nK0DCg2W9vanmbyLaJNcmHi7K3GboX/view?usp=drive_link)
Project repository for CSE 599c taught in the 2024 winter. 

## Abstract
This study proposes a novel Transformer-based approach to audio-to-score alignment (A2SA), diverging from traditional Dynamic Time Warping (DTW) and neural network methods. As opposed to score-to-audio alignment, A2SA treats audio performances as queries to align with reference scores, addressing the variability of live music performance. Our cross-modal alignment pipeline directly generates an alignment score matrix between audio spectrograms and score note-onsets, eliminating the need for preliminary transcription or synthesis. Conducted on the MAESTRO dataset, our experiments assess the Transformer encoders' ability to represent audio and score data in their native modalities, aiming to overcome the limitations of hand-crafted features and prior neural approaches. Despite initial challenges in model convergence, our results suggest promise of attention-based models for A2SA and propose future research directions, including the exploration of positional encoding strategies, loss function adaptations, and architectural adjustments to better capture musical structure.

## Contents
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
