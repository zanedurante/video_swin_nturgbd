#!/bin/bash
python ../Video-Swin-Transformer/tools/analysis/analyze_logs.py plot_curve $@ --keys top1_acc top5_acc 
