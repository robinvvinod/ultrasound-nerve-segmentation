import os 
import sys
import subprocess

!rm -rf /kaggle/working/*
!git clone https://github.com/robinvvinod/ultrasound-nerve-segmentation-keras
!mv ultrasound-nerve-segmentation-keras/* /kaggle/working; rm -rf ultrasound-nerve-segmentation-keras; rm train.py
