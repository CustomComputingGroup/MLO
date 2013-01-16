#!/bin/bash

python mlo.py $@ | grep -v "Segmentation fault"
