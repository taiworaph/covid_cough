#!/usr/bin/env bash

echo This is the working directory of the ec2 instance $(pwd)

#virtual environment
conda update -n base conda --yes
sudo ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
conda create --name covid python=3.7 --yes
conda activate covid

# Python pip package download and deployment
cd ..
cd ..
conda install mysqlclient --yes
echo "y" | pip install -r requirements.txt

#kickstart the  main.py program
cd src/app
echo ---------------------------------------------------
echo ---------------------------------------------------
echo ---------------------------------------------------
echo -----------------Installation Complete-------------
echo ---------------------------------------------------
echo ---------------------------------------------------
echo ---------------------------------------------------

#Ensures keep serving app even after SSH is out of the server.....
nohup python main.py &






