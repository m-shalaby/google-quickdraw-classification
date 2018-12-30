#!/bin/sh

gcc -c reduce.*
gcc -shared -Wl,-soname,reduce.dll -o reduce.dll reduce.o
