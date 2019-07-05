#!/usr/bin/env bash

while true; do
     read -p "Do you wish to download and unzip the pretraining corpus? It is 5GB compressed and 40GB uncompressed? [y/n]" yn
     case $yn in
	    [Yy]* ) wget http://s3.amazonaws.com/proteindata/data/pfam.tar.gz; tar -xzf pfam.tar.gz -C ./data; rm pfam.tar.gz; break;;
            [Nn]* ) exit;;
	    * ) echo "Please answer yes (Y/y) or no (N/n).";;
    esac
done

wget http://s3.amazonaws.com/proteindata/data/secondary_structure.tar.gz
wget http://s3.amazonaws.com/proteindata/data/proteinnet.tar.gz
wget http://s3.amazonaws.com/proteindata/data/remote_homology.tar.gz
wget http://s3.amazonaws.com/proteindata/data/fluorescence.tar.gz
wget http://s3.amazonaws.com/proteindata/data/stability.tar.gz

mkdir -p ./data

tar -xzf secondary_structure.tar.gz -C ./data
tar -xzf proteinnet.tar.gz -C ./data
tar -xzf remote_homology.tar.gz -C ./data
tar -xzf fluorescence.tar.gz -C ./data
tar -xzf stability.tar.gz -C ./data

rm secondary_structure.tar.gz
rm proteinnet.tar.gz
rm remote_homology.tar.gz
rm fluorescence.tar.gz
rm stability.tar.gz
