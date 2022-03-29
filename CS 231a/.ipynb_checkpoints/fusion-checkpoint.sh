# Usage: bash fusion.sh [cfg_file1] [chkpt1] [cfg_file2] [chkpt2]
python ../Video-Swin-Transformer/tools/test.py $1 $2 --out result_1.json
python ../Video-Swin-Transformer/tools/test.py $3 $4 --out result_2.json
python fusion.py result_1.json result_2.json
