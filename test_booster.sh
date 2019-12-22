#!/bin/bash
for file in data/*
do
    echo "Start testing for $file"
    python3 test_classifier.py --dir-path $file
done
