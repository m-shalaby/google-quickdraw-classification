#include <stdio.h>
#include <stdlib.h>
#include "reduce.h"

int main(int argc, const char * argv[]) {

    unsigned char * image = (unsigned char *) malloc(100 * 100 * sizeof(unsigned char));
    if (!image) abort();
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-result"
    for (int i = 0; i < 10000; ++i) scanf("%hhu", &image[i]);
    #pragma GCC diagnostic pop
    reduceImageArray(image);
    for (int i = 0; i < 10000; ++i) { printf("%c ", image[i] ? '1' : '0'); }
    printf("\n");
    free(image);
    return 0;

}
