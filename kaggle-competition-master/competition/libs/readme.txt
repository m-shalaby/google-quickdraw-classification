Make sure that you have g++ for x86_64 to compile the test dll for python
This comes from https://stackoverflow.com/questions/145270/calling-c-c-from-python

run this in the commmand prompt while in the libs folder to build a new dll

g++ -c -fPIC foo.cpp -o foo.o
g++ -shared -Wl,-soname,libfoo.dll -o libfoo.dll  foo.o



g++ -c -fPIC preprocessing.cpp -o preprocessing.o
g++ -shared -Wl,-soname,preprocessing.dll -o preprocessing.dll preprocessing.o