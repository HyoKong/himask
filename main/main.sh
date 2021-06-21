#!/bin/bash

cfg=../exp/cdcn_jigsaw1_multistep_e80.yaml
# train
horovodrun -np 4 -H localhost:4 python -u train.py --cfg $cfg || exit
# eval
python -u eval.py --cfg $cfg --epoch 0 --gpu 0
