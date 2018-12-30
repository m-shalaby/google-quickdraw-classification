#ifndef __REDUCE_H__
#define __REDUCE_H__

#define BOUNDING_BOX_REMOVAL_ITERATIONS 0
#define CLIPPING_BOUNDARY_VALUE 0x80
#define FRAME_IDENTIFICATION_ITERATIONS 0x0200
#define FRAME_SHRINK_RATIO_VALUE 0.0
#define MAXIMUM_BOUNDING_BOX_AREA 0
#define MINIMUM_BOUNDING_BOX_AREA 0
#define MINIMUM_VOID_BOUNDING_ZONE 0
#define PIXEL_BOUNDING_RADIUS 0
#define PIXELS_COUNT_DIFFERENCE_RATIO_VALUE 1.0
#define PIXEL_REMOVAL_ITERATIONS 0
#define PIXEL_REMOVAL_RATIO_VALUE 0.0
#define VERBOSE_MODE_ENABLED

struct Frame {

    int columnsCount, rowsCount;
    int columnsOffset, rowsOffset;

};

struct Image {

    struct Frame imageFrame;
    int columnsCount, rowsCount;
    unsigned char * restrict * restrict imageMatrix;

};

unsigned char * reduceImageArray(unsigned char * restrict);

/*
 *
 *  reduceImage(struct Image *) -> int
 *  
 *  input: reference to a bitmap image structure in row-major form
 *  output: the number of pixels removed from the image structure
 * 
 *  reduceImage performs various transformations on the image matrix to identify the main sub-frame
 *  the identified offsets and side lengths are stored in the imageFrame structure of the function input
 *  transformation parameters are defined as mutable preprocessor macros for reduction strength
 * 
 *  warning: function may enter an infinite loop or access illegal memory regions depending on macro values
 *  warning: return value is an over-estimate of actual pixels removed due to lack of thread synchronization
 * 
 */

int reduceImage(struct Image * restrict);

#endif
