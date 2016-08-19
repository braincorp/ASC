# Adaptive Sparse Coding Cortical Model

## Introduction
This repository contains the core code used for the model published in the paper "Fundamental principles of cortical computation: unsupervised learning with prediction, compression and feedback".

## Starting up


Make sure that virtualenv is installed on your system first.
Then clone the repo and run the following commands:

```
git clone git@github.com:braincorp/ASC
cd ASC
./setup.sh

source venv/bin/activate
python ASC/train_sparse_coding.py [list of video files] [total number of frames]

```

