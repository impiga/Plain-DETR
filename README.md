# Plain-DETR

By [Yutong Lin](https://github.com/impiga)\*, [Yuhui Yuan](https://github.com/PkuRainBow)\*, [Zheng Zhang](https://stupidzz.github.io/)\*, [Chen Li](https://github.com/LC-Edward), [Nanning Zheng](http://www.iair.xjtu.edu.cn/info/1046/1229.htm) and [Han Hu](https://ancientmooner.github.io/)\*

This repo is the official implementation of "[DETR Doesn’t Need Multi-Scale or Locality Design](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_DETR_Does_Not_Need_Multi-Scale_or_Locality_Design_ICCV_2023_paper.html)".

## Introduction

We present an improved DETR detector that maintains a “plain” nature: using a single-scale feature map and global cross-attention calculations without specific locality constraints, in contrast to previous leading DETR-based detectors that re-introduce architectural inductive biases of multi-scale and locality into the decoder.

We show that two simple technologies are surprisingly effective within a plain design: 1) a box-to-pixel relative position bias (BoxRPB) term to guide each query to attend to the corresponding object region; 2) masked image modeling (MIM)-based backbone pre-training to help learn representation with fine-grained localization ability and to remedy dependencies on the multi-scale feature maps.

## Main Results
| BoxRPB | MIM PT. | Reparam. | AP | Paper Position | CFG | CKPT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| ✗ | ✗ | ✗ | 37.2 | Tab2 Exp1 | [cfg](./configs/swinv2_small_sup_pt_ape.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_sup_pt_ape.pth?sv=2020-04-08&st=2023-11-11T09%3A53%3A39Z&se=2049-12-31T09%3A53%3A00Z&sr=b&sp=r&sig=1ntSPDFHXBDVfYts8aUX8xTxA2kRpZyEQdwZL0tRNSk%3D)
| ✓ | ✗ | ✗ | 46.1 | Tab2 Exp2 | [cfg](./configs/swinv2_small_sup_pt_boxrpe.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_sup_pt_boxrpe.pth?sv=2020-04-08&st=2023-11-11T09%3A54%3A10Z&se=2023-11-12T09%3A54%3A10Z&sr=b&sp=r&sig=yg4gw7vWX8zlOurS3x2J9%2BPwfsSaHEYYOHE4DJTWw%2BQ%3D) 
| ✓ | ✓ | ✗ | 48.7 | Tab2 Exp5 | [cfg](./configs/swinv2_small_mim_pt_boxrpe.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_mim_pt_boxrpe.pth?sv=2020-04-08&st=2023-11-11T09%3A52%3A38Z&se=2049-12-31T09%3A52%3A00Z&sr=b&sp=r&sig=eX%2FNgca78ccyBhlujtCSh1BDHiPPOjjceyrMMLxKgr8%3D)
| ✓ | ✓ | ✓ | 50.9 | Tab2 Exp6 | [cfg](./configs/swinv2_small_mim_pt_boxrpe_reparam.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_mim_pt_boxrpe_reparam.pth?sv=2020-04-08&st=2023-11-14T07%3A42%3A25Z&se=2049-12-31T07%3A42%3A00Z&sr=b&sp=r&sig=5r09k4tFNIO%2FURYIQ2RbjJOU7v4dWqFW1D3F%2Bdg%2FYq0%3D)

## Installation

### Conda

```
# create conda environment
conda create -n plain_detr python=3.8 -y
conda activate plain_detr

# install pytorch (other versions may also work)
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# other requirements
git clone https://github.com/impiga/Plain-DETR.git
cd Plain-DETR
pip install -r requirements.txt
```

### Docker

We have tested with the docker image `superbench/dev:cuda11.8`. Other dockers may also work.

```
# run docker
sudo docker run -it -p 8022:22 -d --name=plain_detr --privileged --net=host --ipc=host --gpus=all -v /:/data superbench/dev:cuda11.8 bash
sudo docker exec -it plain_detr bash

# other requirements
git clone https://github.com/impiga/Plain-DETR.git
cd Plain-DETR
pip install -r requirements.txt
```

## Usage

### Dataset preparation

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### Pretrained models preparation

Please run the following script to download supervised and mask-image-modeling pretrained models.

(We adopts Swin Transformer v2 as the default backbone. If you are interested in the pretraining, please refer to Swin Transformer v2 ([paper](https://arxiv.org/abs/2111.09883), [github](https://github.com/microsoft/Swin-Transformer)) and SimMIM ([paper](https://arxiv.org/abs/2111.09886), [github](https://github.com/microsoft/SimMIM)) for more details.)

```
bash tools/prepare_pt_model.sh
```

### Training

#### Training on single node

```
GPUS_PER_NODE=<num gpus> ./tools/run_dist_launch.sh <num gpus> <path to config file>
```

#### Training on multiple nodes

On each node, run the following script:

```
MASTER_ADDR=<master node IP address> GPUS_PER_NODE=<num gpus> NODE_RANK=<rank> ./tools/run_dist_launch.sh <num gpus> <path to config file> 
```

### Evaluation

To evalute a plain-detr model, please run the following script:

```
 <path to config file> --eval --resume <path to plain-detr model>
```

You could also use `./tools/run_dist_launch.sh` to evaluate a model on multiple GPUs.

## Limitation & Discussion
 - While we have eliminated multi-scale designs for the backbone output and decoder input, the generation of proposals still depends on multi-scale features.

   We have performed trials utilizing single-scale features for proposals(not included in the paper), but it led to ~1 mAP performance drop.

## Known issues

- Most of our experiments are conducted on 16 GPUs with 1 image per GPU. We have tested our released checkpoints with larger batch size and found that the performance of first three models drops significantly.

  We are now reviewing our implementation and will update our code to support larger batch size for both training and inference.

## Citing Plain-DETR

If you find Plain-DETR useful in your research, please consider citing:
```
inproceedings{lin2023detr,
  title={DETR Does Not Need Multi-Scale or Locality Design},
  author={Lin, Yutong and Yuan, Yuhui and Zhang, Zheng and Li, Chen and Zheng, Nanning and Hu, Han},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6545--6554},
  year={2023}
}
```