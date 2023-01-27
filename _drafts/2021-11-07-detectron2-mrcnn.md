---
layout: post
title:  "Walkthrough of Detectron MRCNN"
date:   2021-11-07 06:04:00 +0100
categories: jekyll update
---

In this post we will go through how Mask-RCNN is implemented in the Detectron framework. We will see what happens from the moment  we `cd` into `detectron2/tools` and run the following command

```bash
./train_net.py --num-gpus 8 \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
```

With `train_net.py` the `config-file` and `num-gpus` arguments are passed into the `launch` function from `detectron2.engine.launch` along with other default arguments.

The purpose of `launch` is to run the `main` function either on a single machine or in a distributed manner as indicated in the command line arguments. The work of distributing the training takes place in `_distributed_worker`.  

We will come back and look at how distributed training happens.

But first let us go through the steps in `main`. Before anything happens there is a setup step (`cfg = setup(args)`).

Here a default config file is updated (`cfg.merge_from_file(args.config_file)`) based on the experiment config file which here is as follows

```
_BASE_: "../Base-RCNN-FPN.yaml"
MODEL:
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
  MASK_ON: True
  RESNETS:
    DEPTH: 50
```

If config values are overridden in args these will also be updated. 

For a training run `args.eval_only` is `False` so first a `Trainer` instance is created.  






