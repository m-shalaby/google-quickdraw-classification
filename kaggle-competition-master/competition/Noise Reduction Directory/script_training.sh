#!/bin/bash

gcc -Ofast *.c *.h
for i in `seq 0 999`; do python test_test.py $i | ./a.out >> test_features_0_999.txt; echo $i; done &
for i in `seq 1000 1999`; do python test_test.py $i | ./a.out >> test_features_1000_1999.txt; echo $i; done &
for i in `seq 2000 2999`; do python test_test.py $i | ./a.out >> test_features_2000_2999.txt; echo $i; done &
for i in `seq 3000 3999`; do python test_test.py $i | ./a.out >> test_features_3000_3999.txt; echo $i; done &
for i in `seq 4000 4999`; do python test_test.py $i | ./a.out >> test_features_4000_4999.txt; echo $i; done &
for i in `seq 5000 5999`; do python test_test.py $i | ./a.out >> test_features_5000_5999.txt; echo $i; done &
for i in `seq 6000 6999`; do python test_test.py $i | ./a.out >> test_features_6000_6999.txt; echo $i; done &
for i in `seq 7000 7999`; do python test_test.py $i | ./a.out >> test_features_7000_7999.txt; echo $i; done &
for i in `seq 8000 8999`; do python test_test.py $i | ./a.out >> test_features_8000_8999.txt; echo $i; done &
for i in `seq 9000 9999`; do python test_test.py $i | ./a.out >> test_features_9000_9999.txt; echo $i; done &
wait
cat test_features_0_999.txt test_features_1000_1999.txt test_features_2000_2999.txt test_features_3000_3999.txt test_features_4000_4999.txt test_features_5000_5999.txt test_features_6000_6999.txt test_features_7000_7999.txt test_features_8000_8999.txt test_features_9000_9999.txt > test_features.txt
for i in `seq 0 999`; do python test_train.py $i | ./a.out >> train_features_0_999.txt; echo $i; done &
for i in `seq 1000 1999`; do python test_train.py $i | ./a.out >> train_features_1000_1999.txt; echo $i; done &
for i in `seq 2000 2999`; do python test_train.py $i | ./a.out >> train_features_2000_2999.txt; echo $i; done &
for i in `seq 3000 3999`; do python test_train.py $i | ./a.out >> train_features_3000_3999.txt; echo $i; done &
for i in `seq 4000 4999`; do python test_train.py $i | ./a.out >> train_features_4000_4999.txt; echo $i; done &
for i in `seq 5000 5999`; do python test_train.py $i | ./a.out >> train_features_5000_5999.txt; echo $i; done &
for i in `seq 6000 6999`; do python test_train.py $i | ./a.out >> train_features_6000_6999.txt; echo $i; done &
for i in `seq 7000 7999`; do python test_train.py $i | ./a.out >> train_features_7000_7999.txt; echo $i; done &
for i in `seq 8000 8999`; do python test_train.py $i | ./a.out >> train_features_8000_8999.txt; echo $i; done &
for i in `seq 9000 9999`; do python test_train.py $i | ./a.out >> train_features_9000_9999.txt; echo $i; done &
wait
cat train_features_0_999.txt train_features_1000_1999.txt train_features_2000_2999.txt train_features_3000_3999.txt train_features_4000_4999.txt train_features_5000_5999.txt train_features_6000_6999.txt train_features_7000_7999.txt train_features_8000_8999.txt train_features_9000_9999.txt > train_features.txt
rm a.out
rm reduce.h.gch
python classification.py
