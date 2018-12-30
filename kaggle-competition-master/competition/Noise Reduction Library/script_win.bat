gcc -c reduce.h reduce.c
gcc -shared -Wl,-soname,reduce.dll -o reduce.dll reduce.o
pause