#!/bin/bash
export PYTHONPATH=libs/pyXGPR/src:libs/HTML.py-0.04
python optimizer.py $@ | grep -v "Segmentation fault"
