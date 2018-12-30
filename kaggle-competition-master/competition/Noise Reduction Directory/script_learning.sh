#!/bin/bash

gcc -Ofast *.c *.h
for i in `seq 0 49`; do python test_train.py $i | ./a.out; done
rm a.out
rm reduce.h.gch