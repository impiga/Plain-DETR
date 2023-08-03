# Plain-DETR

By [Yutong Lin](https://github.com/impiga)\*, [Yuhui Yuan](https://github.com/PkuRainBow)\*, [Zheng Zhang](https://stupidzz.github.io/)\*, [Chen Li](https://github.com/LC-Edward), [Nanning Zheng](http://www.iair.xjtu.edu.cn/info/1046/1229.htm) and [Han Hu](https://ancientmooner.github.io/)\*

This repo is the official implementation of "DETR Doesn’t Need Multi-Scale or Locality Design".

## Introduction

We present an improved DETR detector that maintains a “plain” nature: using a single-scale feature map and global cross-attention calculations without specific locality constraints, in contrast to previous leading DETR-based detectors that re-introduce architectural inductive biases of multi-scale and locality into the decoder.

We show that two simple technologies are surprisingly effective within a plain design: 1) a box-to-pixel relative position bias (BoxRPB) term to guide each query to attend to the corresponding object region; 2) masked image modeling (MIM)-based backbone pre-training to help learn representation with fine-grained localization ability and to remedy dependencies on the multi-scale feature maps. 

(Code and model weights will be available in the near future.)
