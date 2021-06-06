#!/bin/sh
python train.py \
--img 512 \
--batch 8 \
--epochs 10 \
--data "./fold0/data.yaml" \
--cfg "./models/yolov4-p7.yaml" \
--weights "" \
--name yolov4-p7-debug \
--device 6,7