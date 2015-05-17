#!/bin/sh


### Только нумерикал фича
python main.py --algorithm sg \
               --add-bias \
               --weights-storage main.DictStorage \
               --rate ada_grad \
               --rate-ada-grad-beta 1 \
               --rate-ada-grad-alpha 0.1 \
               --feature-filter none \
               --subsampling \
               --subsampling-label 0 \
               --subsampling-rate 0.35 \
               --progressive-validation \
               --progressive-validation-depth 100000 \
               --storage-path /home/ilariia/CTR_models \
               --storage-label num01 \
               --storage-metrics-dumping-depth 100000 \
               --missing-plain-features online_average