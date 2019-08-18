#!/bin/sh
mkdir data
cd data
wget http://csr.bu.edu/ftp/visda/2019/semi-supervised/real.zip -O real.zip
wget http://csr.bu.edu/ftp/visda/2019/semi-supervised/sketch.zip -O sketch.zip
wget http://csr.bu.edu/ftp/visda/2019/semi-supervised/txt.zip -O txt.zip
unzip real.zip
unzip sketch.zip
unzip txt.zip

