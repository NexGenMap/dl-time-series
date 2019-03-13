#!/bin/bash

echo "Setup OS packages"

sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get -y update
sudo apt install -y git wget python3 python3-dev python3-gdal gdal-bin libgdal-dev
sudo apt upgrade -y git wget python3 python3-dev python3-gdal gdal-bin libgdal-dev

echo "Setup Python dependencies packages"

sudo pip3 install numpy
sudo pip3 install tensorflow-gpu==1.7.0 
sudo pip3 install keras
sudo pip3 install scikit-learn 
sudo pip3 install download https://bitbucket.org/chchrsc/rios/downloads/rios-1.4.5.tar.gz

echo "Installation Done"