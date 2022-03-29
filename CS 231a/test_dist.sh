# Usage: bash test.sh path/to/config/file.py path/to/chkpt/file.pth NUM_GPUS output/pth.txt
bash ../Video-Swin-Transformer/tools/dist_test.sh $1 $2 $3 --out $4
