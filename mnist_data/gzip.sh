#! /bin/sh
rm -rf unzip_file
mkdir unzip_file
gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
mv train-images-idx3-ubyte unzip_file
mv train-labels-idx1-ubyte unzip_file
mv t10k-images-idx3-ubyte unzip_file
mv t10k-labels-idx1-ubyte unzip_file