#!/bin/bash
export PYTHONPATH=libs/pyXGPR/src
python mlo.py $@ | grep -v "Segmentation fault"
