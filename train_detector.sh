#!/bin/bash
for file in data/*
do
    echo "Start Learning for $file"
    python3 train_autoenc.py --dir-path $file --epochs 200 --lr 0.1
    python3 train_autoenc.py --dir-path $file --epochs 300 --lr 0.1 --fisher 5
    python3 train_autoenc.py --dir-path $file --epochs 300 --lr 0.1 --fisher 10
done
