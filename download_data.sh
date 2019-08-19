#!/bin/sh
mkdir data data/multi
cd data/multi
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip -O real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip -O sketch.zip
unzip real.zip
unzip sketch.zip

