# HiFaceMask Challenge

## Brief Intro of Our Method

- Backbone: CDCN (resnet-like with attention mechanism)
- Output: 
  - Parsing map (real face region --> 1; background and fake face region --> 0)
  - Cross-Entropy Classification

## Dataset and Model Origanization
```
database/
├── test
│   └── 0000
├── val
│   └── 0000
├── Parsing
│   ├── 1_06_0_1_1_1
│   │   └── *.png
│   └── 1_06_0_1_1_2
│       └── *.png
├── train
│   └── 0000
├── himask_test.json
├── himask.json
└── train_label.txt

output/
└── model_dump
    └── *.h5 (Please put the pre-trained model in this folder.)

```
The json files and the parsing maps can be downloaded from: [here](https://drive.google.com/drive/folders/1mVCgaUKAU64lEshzTZSznSZL1cbqg77n?usp=sharing).


## Train & Eval
```
cd main
sh main.sh
```

If you want to do inference directly, please comment out the following in main.sh:

```
horovodrun -np 4 -H localhost:4 python -u train.py --cfg $cfg || exit
```

## Misc

If you have any questions about the repo, please feel free to reach out via hanyang.k(at)u(dot)nus(dot)edu or hyokong(at)stu(dot)xjtu(dot)edu(dot)cn.

