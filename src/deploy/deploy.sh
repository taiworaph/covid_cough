#!/usr/bin/env bash

echo This is the working directory of the ec2 instance $(pwd)

#virtual environment
conda create --name covid python=3.7 --yes
conda activate covid

# Python pip package download and deployment
cd ..
conda install mysqlclient --yes
echo "y" | pip install -r ../src/app/requirements.txt

echo ---------------------------------------------------
echo -----------------Installation Complete-------------
echo ---------------------------------------------------




