#!/bin/bash
export PYTHONPATH=libs/pyXGPR/src
python2.7 optimizer.py $@ | grep -v "Segmentation fault"
