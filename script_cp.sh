#!/bin/bash

#Ensemble size
N=3


for i in `seq 2 $N` 
do
#echo $i
cp -r member_1 member_$i
done
