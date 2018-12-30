#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The following code was written by Yi Feng Yan and adapted by Samuel Hatin to work with python


#define NOISE_RATIO_LIMIT    0.0625
#define PIXEL_RADIUS_SIZE    0x0008
#define REDUCTION_ITERATIONS 0x1000

// use an array of 128 SIMD vectors of size 128 bytes

static __attribute__((__vector_size__(sizeof(char) * 128))) char canvas[128] = {0};

//  declarations for canvas transform operations useful for preprocessing

static inline int getRandomNumber(void);
static inline void flipCanvasHorizontally(void);
static inline void printCanvas(void);
static inline void reduceNoise(void);

//  takes path to data file and image index as launch parameters

int main(int argc, const char * argv[]) {

    //  opens data file for reading and updating at appropriate position

    FILE * dataFile;
    if (argc != 3 || !(dataFile = fopen(argv[1], "r+"))) { exit(1); }
    long int fileOffsetPosition = atoi(argv[2]) * (1250 + 1);
    fseek(dataFile, fileOffsetPosition, SEEK_SET);

    //  loads image pixels data into 2D vector for preprocessing

    char bufferedData[1250];
    if (fread(bufferedData, 1250, 1, dataFile) < 1) { exit(1); }
    for (int i = 0; i < 10000; ++i) { canvas[i / 100][i % 100] = (bufferedData[i / 8] & (0b1 << (7 - (i % 8)))) ? 1 : 0; }

    //  preprocesses canvas using custom function calls

    printf("---------- Default Layout ----------\n");
    printCanvas();
    printf("---------- Flipped Layout ----------\n");
    flipCanvasHorizontally();
    printCanvas();
    flipCanvasHorizontally();
    printf("---------- Reduced Layout ----------\n");
    reduceNoise();
    printCanvas();

    //  applies contents of 2D vector back into data file (comment out to avoid permanent changes)

    // memset(bufferedData, 0, 1250);
    // for (int i = 0; i < 10000; ++i) { bufferedData[i / 8] |= canvas[i / 100][i % 100] ? 0b1 << (7 - (i % 8)) : 0; }
    // fseek(dataFile, fileOffsetPosition, SEEK_SET);
    // if (fwrite(bufferedData, 1250, 1, dataFile) < 1) { exit(1); }
    fclose(dataFile);
    return 0;

}

//  returns randomly generated number (not thread-safe, but faster than default blocking rand)

static inline int getRandomNumber(void) {

    static unsigned int seed = 2531011;
    seed = 214013 * seed + 2531011;
    return (seed >> 16) & 0x7FFF;

}

//  examples of how preprocessing can be achieved

static inline void flipCanvasHorizontally(void) {

    #pragma omp parallel for
    for (int i = 0; i < 50; ++i) {

        int k = 100 - i - 1;
        canvas[i] ^= canvas[k];
        canvas[k] ^= canvas[i];
        canvas[i] ^= canvas[k];

}   }

// 	print canvas to console

static inline void printCanvas(void) {

    for (int i = 0; i < 100; ++i) {

        for (int j = 0; j < 100; ++j) printf("%d", canvas[i][j]);
        printf("\n");

}   }

//  eliminates sorely isolated pixels heuristically with no guarantee

static inline void reduceNoise(void) {

    #pragma omp parallel for
    for (int i = 0; i < REDUCTION_ITERATIONS; ++i) {

        int column, row;				// Pixel location on the canvas
		int counter = 0;				// number of black pixel in the bounding box
		int bottom, left, right, top;	// Bounding box around the pixel
		
		__attribute__((__vector_size__(sizeof(char) * 128))) char segment = {0};	//segment is a SIMD vector
		
		// Pick random pixel with value 1
        do { column = getRandomNumber() % 100, row = getRandomNumber() % 100; } while (!canvas[row][column]);
		
		// Bounding box around (column, row)
		
        bottom 	= row 		+ PIXEL_RADIUS_SIZE;
        left 	= column 	- PIXEL_RADIUS_SIZE;
        right 	= column 	+ PIXEL_RADIUS_SIZE;
        top 	= row    	- PIXEL_RADIUS_SIZE;
			   
        bottom = bottom >= 100 ? 99 : bottom;
        left = left < 0 ? 0 : left;
        right = right >= 100 ? 99 : right;
        top = top < 0 ? 0 : top;
		
		// traverse each segment (horizontal line of the bounding box) and sum it up in the segment
        for (int offset = top; offset <= bottom; ++offset) 
			segment += canvas[offset];
		// traverse the segment (within the bounding box) and sum up the value of the pixels
        for (int offset = left; offset <= right; ++offset) 
			counter += segment[offset];
		
		// if the ratio of colored points versus the total number of points exceeds threshold, set to 0 else set to 1.
        canvas[row][column] = (float) counter / ((bottom - top + 1) * (right - left + 1)) > NOISE_RATIO_LIMIT ? 1 : 0;

}   }


// 	Allows python to call these functions

extern "C" {
	//example:
    int getRandomNumberPy(){
		return getRandomNumber(); 
	}
	
}