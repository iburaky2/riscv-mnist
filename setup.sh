#!/bin/bash
GREEN='\e[32m'
NC='\e[0m'

echo -e "${GREEN}Downloading the MNIST database to data/${NC}"
echo ""
mkdir -p data
cd data
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

echo -e "${GREEN}Unzipping the database${NC}"
echo ""
gunzip -f *.gz
cd ..

echo -e "${GREEN}Running python script to extract test images and labels & train the model${NC}"
cd scripts
python3 extract_and_train.py
cd ..

echo -e "${GREEN}Compiling firmware${NC}"
cd firmware
make