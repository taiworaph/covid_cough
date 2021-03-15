#!/usr/bin/env bash

echo This is the working directory of the ec2 instance $(pwd)
echo "Please ensure you are in the conda environment with the relevant packages for running this algorithm"
echo "Please ensure all models to be evaluated are in the data directory"
echo "Please enter the cur-off value to be utilized for evaluating model performance"
read cut_off


python automated_model_testing.py ../data $cut_off

echo "--------------------------------------------"
echo "-----------------Completed------------------"
echo "--------------------------------------------"
