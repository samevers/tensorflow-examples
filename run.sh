#! /bin/bash
############################################
#
# Author: 
# E-Mail:@sogou-inc.com
# Create time: 2017 4�� 20 11ʱ57��40��
# version 1.0
#
############################################
nvidia-smi |fgrep Default|awk 'BEGIN{srand()}{
print (NR-1)"\t"substr($13,1,length($13)-1)"\t"rand()
}'|sort -t$'\t' -k 2,2n -k 3,3n|awk -F"\t" 'NR==1{
print "export CUDA_VISIBLE_DEVICES="$1
print "echo $CUDA_VISIBLE_DEVICES"
}' > setdevicesub.sh
. setdevicesub.sh
rm -f setdevicesub.sh

#python t.py
#python testlinear.py
#python testlr.py
#python autoencoder.py
#python multilayer_perceptron.py 1>std 2>err
python rnn_networks.py
