#!/bin/sh

cat /home/ilariia/ads/train_header.txt /home/ilariia/ads/train_shuffled.txt | ciw learn \
    --algorithm sg \
    --add-bias \
    --weights-storage ciw.main.DictStorage \
    --rate ada_grad \
    --rate-ada-grad-beta 1 \
    --rate-ada-grad-alpha 1 \
    --feature-filter none \
    --progressive-validation \
    --progressive-validation-depth 100000 \
    --storage-metrics-dumping-depth 100000 \
    --storage-model-dumping-depth 1000000 \
    --missing-plain-features average \
    --normalize-plain-features \
    --storage-path /home/ilariia/CTR_models \
    --storage-label all

cat /home/ilariia/ads/validate.txt | ciw validate \
    --storage-path /home/ilariia/CTR_models \
    --storage-label all \
    --storage-predictions-id validations8.6 \
    --storage-predictions-dumping-depth 100000

cat /home/ilariia/ads/test.txt | ciw predict \
    --storage-path /home/ilariia/CTR_models \
    --storage-label all \
    --storage-predictions-id predictions
