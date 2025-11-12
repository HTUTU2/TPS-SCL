# TPS-SCL

Code for our paper entilted "Breaking Alignment Barriers: TPS-Driven Semantic Correlation Learning for Alignment-Free
RGB-T Salient Object Detection" ,accepted at AAAI 2026.

Thank you for your attention.

## Method

![Method](https://raw.githubusercontent.com/HTUTU2/TPS-SCL/refs/heads/main/assets/frame.png)

### Requirement

torch 2.4.0

torchaudio 2.4.0

torchvision 0.19.0

timm 1.0.20

mamba-ssm 2.0.3

causal-conv1d 1.4.0

## Dataset

For dataset acquisition, please refer to [PCNet](https://github.com/Angknpng/PCNet).

## Ablation Studies

| Method | BackBone |           | Component |          |          | Repalce TPS     |          | UVT20K |       |        | UVT2000 |        |        |
|--------|----------|-----------|-----------|----------|----------|-----------------|----------|--------|-------|--------|---------|--------|--------|
| ID     | SwinB    | Res2Net50 | SCCM      | TPSAM    | CMCM     | Cross Attention | DCNv     | F↑     | S↑    | E↑     | F↑      | S↑     | E↑     |
| 1      | &#10004; |           |           |          |          |                 |          | 0.577  | 0.694 | 0.679  | 0.416   | 0.632  | 0.614  |
| 2      | &#10004; |           | &#10004;  |          |          |                 |          | 0.621  | 0.784 | 0.756  | 0.489   | 0.725  | 0.678  |
| 3      | &#10004; |           |           | &#10004; |          |                 |          | 0.614  | 0.775 | 0.749  | 0.492   | 0.697  | 0.671  |
| 4      | &#10004; |           |           |          | &#10004; |                 |          | 0.599  | 0.763 | 0.712  | 0.465   | 0.705  | 0.663  |
| 5      | &#10004; |           | &#10004;  | &#10004; |          |                 |          | 0.763  | 0.804 | 0.831  | 0.56    | 0.71   | 0.684  |
| 6      | &#10004; |           |           | &#10004; | &#10004; |                 |          | 0.022  | 0.431 | 0.516  | 0.024   | 0.465  | 0.625  |
| 7      | &#10004; |           | &#10004;  |          | &#10004; |                 |          | 0.625  | 0.792 | 0.763  | 0.498   | 0.735  | 0.707  |
| 8      | &#10004; |           | &#10004;  |          | &#10004; | &#10004;        |          | 0.7997 | 0.851 | 0.872  | 0.6107  | 0.762  | 0.757  |
| 9      | &#10004; |           | &#10004;  |          | &#10004; |                 | &#10004; | 0.7707 | 0.844 | 0.861  | 0.628   | 0.783  | 0.788  |
| 10     | &#10004; |           | &#10004;  | &#10004; | &#10004; |                 |          | 0.815  | 0.866 | 0.887  | 0.632   | 0.794  | 0.792  |
| 11     |          | &#10004;  | &#10004;  | &#10004; | &#10004; |                 |          | 0.7568 | 0.831 | 0.8562 | 0.5898  | 0.7741 | 0.7772 |

## Citation

If you think our work is helpful, please cite:

```
@inproceedings{hlp2026alignment,
  title={Breaking Alignment Barriers: TPS-Driven Semantic Correlation Learning for Alignment-Free RGB-T Salient Object Detection},
  author = {Hu, Lupiao and Wang, Fasheng and Chen, Fangmei and Sun, Fuming and Li, Haojie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={},
  number={},
  pages={},
  year={2026}
}