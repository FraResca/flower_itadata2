conda env create -f flower.yml

conda activate flower

python3 flserver.py

python3 flclient.py