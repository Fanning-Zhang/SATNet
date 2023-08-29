# SATNet
This repo is the official code of [paper](https://arxiv.org/abs/2308.04156):

**"Towards Top-Down Stereoscopic Image Quality Assessment via Stereo Attention"**.

[Huilin Zhang](www.fanningzhang.tech), Sumei Li, Yongli Chang.

Tianjin University, Tianjin, China.

## To-Do List
- [ ] Release the code of SATNet (satnet.py) after the paper is accepted.


## Requirements
- Python 3.8.5
- PyTorch 1.11.0
- torchvision 0.12.0
- CUDA 11.3

In addition, [requirement.txt](./requirements.txt) lists all the required packages:
```
pip install -r requirements.txt
```


## Demo
We provide a demo to show how to use SATNet to predict the quality of a stereoscopic image pair.

The code is coming soon.


## Datasets
| Datasets | Link |
| --- | --- |
| LIVE 3D Phase I |[Available here](https://live.ece.utexas.edu/research/quality/live_3dimage_phase1.html)|
| LIVE 3D Phase II |[Available here](https://live.ece.utexas.edu/research/quality/live_3dimage_phase2.html)|
| WIVC 3D Phase I & II |[Available here](https://ivc.uwaterloo.ca/database/3DIQA.html)|


## Training
The code is coming soon.


## Citation
If you find this repo helpful, please cite our paper:
```
@misc{zhang2023topdown,
      title={Towards Top-Down Stereoscopic Image Quality Assessment via Stereo Attention}, 
      author={Huilin Zhang and Sumei Li and Yongli Chang},
      year={2023},
      eprint={2308.04156},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```