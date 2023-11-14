#!/usr/bin/env bash

set -x

FILE_NAME=$(basename $0)
EXP_DIR=./exps/${FILE_NAME%.*}
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --mixed_selection \
    --look_forward_twice \
    --num_queries_one2one 300 \
    --num_queries_one2many 1500 \
    --k_one2many 6 \
    --lambda_one2many 1.0 \
    --dropout 0.0 \
    --norm_type pre_norm \
    --backbone swin_v2_small_window12to16_2global \
    --drop_path_rate 0.1 \
    --upsample_backbone_output \
    --upsample_stride 16 \
    --num_feature_levels 1 \
    --decoder_type global_rpe_decomp \
    --decoder_rpe_type linear \
    --proposal_feature_levels 4 \
    --proposal_in_stride 16 \
    --pretrained_backbone_path ./pt_models/swinv2_small_1k_500k_mim_pt.pth \
    --epochs 12 \
    --lr_drop 11 \
    --warmup 1000 \
    --lr 2e-4 \
    --use_layerwise_decay \
    --lr_decay_rate 0.9 \
    --weight_decay 0.05 \
    --wd_norm_mult 0.0 \
    ${PY_ARGS}
