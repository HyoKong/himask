#!/bin/bash

#python -u ../pre_processing/mv2shm.py || exit
#cd ..
#rm -r ./database
#ln -s /dev/shm/database ./database
#cd main

cfg=../exp/cdcn_jigsaw1_multistep_e80.yaml
horovodrun -np 4 -H localhost:4 python -u train.py --cfg $cfg || exit

python -u eval.py --cfg $cfg --epoch 0 --gpu 0