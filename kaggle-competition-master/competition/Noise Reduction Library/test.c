#include <stdio.h>
#include <stdlib.h>
#include "reduce.h"

int main(int argc, const char * argv[]) {

    unsigned char * image = (unsigned char *) malloc(100 * 100 * sizeof(unsigned char));
    if (!image) abort();
    for (int i = 0; i < 10000; ++i) { int ignored = scanf("%hhu", &image[i]); }
    for (int i = 0; i < 10000; ++i) image[i] = i % 100 < 20 ? 0 : image[i];
    reduceImageArray(image);
    return 0;

}