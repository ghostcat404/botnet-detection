#!/bin/bash
for file in data/*
do
    echo "$file"
    echo "Start preprocess"
    python3 utils/preprocess_data.py --dir-path $file
    echo "Start training"
    python3 train_classifier.py --dir-path $file
    echo "Start testing"
    python3 test_classifier.py --dir-path $file
done
