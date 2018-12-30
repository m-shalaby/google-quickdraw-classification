#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "reduce.h"

#ifndef BOUNDING_BOX_REMOVAL_ITERATIONS
#define BOUNDING_BOX_REMOVAL_ITERATIONS 0
#endif
#ifndef CLIPPING_BOUNDARY_VALUE
#define CLIPPING_BOUNDARY_VALUE 0
#endif
#ifndef FRAME_IDENTIFICATION_ITERATIONS
#define FRAME_IDENTIFICATION_ITERATIONS 0
#endif
#ifndef FRAME_SHRINK_RATIO_VALUE
#define FRAME_SHRINK_RATIO_VALUE 0.0
#endif
#ifndef MAXIMUM_BOUNDING_BOX_AREA
#define MAXIMUM_BOUNDING_BOX_AREA 0
#endif
#ifndef MINIMUM_BOUNDING_BOX_AREA
#define MINIMUM_BOUNDING_BOX_AREA 0
#endif
#ifndef MINIMUM_VOID_BOUNDING_ZONE
#define MINIMUM_VOID_BOUNDING_ZONE 0
#endif
#ifndef PIXEL_BOUNDING_RADIUS
#define PIXEL_BOUNDING_RADIUS 0
#endif
#ifndef PIXELS_COUNT_DIFFERENCE_RATIO_VALUE
#define PIXELS_COUNT_DIFFERENCE_RATIO_VALUE 1.0
#endif
#ifndef PIXEL_REMOVAL_ITERATIONS
#define PIXEL_REMOVAL_ITERATIONS 0
#endif
#ifndef PIXEL_REMOVAL_RATIO_VALUE
#define PIXEL_REMOVAL_RATIO_VALUE 0.0
#endif

static inline int clipOutOfRangePixels(struct Image * restrict);
static inline int getFrameArea(struct Frame * restrict);
static inline int growBoundingBox(struct Image * restrict, int *, int, int *, int, int *, int, int *, int);
static inline int isContainedWithinFrame(struct Frame * restrict, int, int);
static inline int nextRandomValue(int *);
static inline int removeSmallBoundingBoxes(struct Image * restrict);
static inline int removeIndividualPixels(struct Image * restrict);
static inline int setMainSubFrameBoundingBox(struct Image * restrict);
static inline int shrinkMainFrameBoundingBox(struct Image * restrict);
static inline int shrinkMainFrameFromBottom(struct Image * restrict);
static inline int shrinkMainFrameFromLeft(struct Image * restrict);
static inline int shrinkMainFrameFromRight(struct Image * restrict);
static inline int shrinkMainFrameFromTop(struct Image * restrict);
static inline void printImageMatrix(struct Image * restrict);
static inline void refitImageMatrix(struct Image * restrict);

unsigned char * reduceImageArray(unsigned char * restrict image) {

    size_t itemSize = sizeof(unsigned char);
    struct Image imageContainer;
    imageContainer.columnsCount = 100;
    imageContainer.rowsCount = 100;
    imageContainer.imageMatrix = (unsigned char **) malloc(100 * sizeof(unsigned char *));
    if (!imageContainer.imageMatrix) abort();

    #pragma omp parallel for
    for (int i = 0; i < 100; ++i) {

        imageContainer.imageMatrix[i] = (unsigned char *) malloc(100 * itemSize);
        if (!imageContainer.imageMatrix[i]) abort();
        memcpy(imageContainer.imageMatrix[i], image + 100 * i * itemSize, 100 * itemSize);

    }

    reduceImage(&imageContainer);
    #pragma omp parallel for
    for (int i = 0; i < 100; ++i) memcpy(image + 100 * i * itemSize, imageContainer.imageMatrix[i], 100 * itemSize);
    #pragma omp parallel for
    for (int i = 0; i < 100; ++i) free(imageContainer.imageMatrix[i]);
    free((void *) imageContainer.imageMatrix);
    return image;

}

int reduceImage(struct Image * restrict image) {

    image->imageFrame.columnsCount = image->columnsCount;
    image->imageFrame.columnsOffset = 0;
    image->imageFrame.rowsCount = image->rowsCount;
    image->imageFrame.rowsOffset = 0;
    int pixelsEliminated = 0;
    #ifdef VERBOSE_MODE_ENABLED
    printf("Original Image:\n");
    printImageMatrix(image);
    #endif
    pixelsEliminated += clipOutOfRangePixels(image);
    // pixelsEliminated += shrinkMainFrameBoundingBox(image);
    // pixelsEliminated += removeIndividualPixels(image);
    // pixelsEliminated += shrinkMainFrameBoundingBox(image);
    // pixelsEliminated += removeSmallBoundingBoxes(image);
    // pixelsEliminated += shrinkMainFrameBoundingBox(image);
    pixelsEliminated += setMainSubFrameBoundingBox(image);
    refitImageMatrix(image);
    #ifdef VERBOSE_MODE_ENABLED
    printf("Reduced Image:\n");
    printImageMatrix(image);
    printf("Pixels Eliminated: %04d\n", pixelsEliminated);
    #endif
    return pixelsEliminated;

}

static inline int clipOutOfRangePixels(struct Image * restrict image) {

    int pixelsEliminated = 0;

    #pragma omp parallel for
    for (int i = 0; i < image->rowsCount; ++i)
    for (int j = 0; j < image->columnsCount; ++j)
    if (image->imageMatrix[i][j] && image->imageMatrix[i][j] < CLIPPING_BOUNDARY_VALUE) {

        image->imageMatrix[i][j] = 0;

        #pragma omp atomic
        pixelsEliminated += 1;

    }

    return pixelsEliminated;

}

static inline int getFrameArea(struct Frame * restrict frame) { return frame->columnsCount * frame->rowsCount; }

static inline int growBoundingBox(struct Image * restrict image,
                                  int * bottom, int bottomBound,
                                  int * left, int leftBound,
                                  int * right, int rightBound,
                                  int * top, int topBound) {

    int pixelsCount = 0;
    if ((*bottom)++ < bottomBound)
    for (int offset = *left; offset <= *right; ++offset) pixelsCount += image->imageMatrix[*bottom][offset] ? 1 : 0;
    if (pixelsCount) return pixelsCount; else *bottom -= 1;
    if ((*left)-- > leftBound)
    for (int offset = *top; offset <= *bottom; ++offset) pixelsCount += image->imageMatrix[offset][*left] ? 1 : 0;
    if (pixelsCount) return pixelsCount; else *left += 1;
    if ((*right)++ < rightBound)
    for (int offset = *top; offset <= *bottom; ++offset) pixelsCount += image->imageMatrix[offset][*right] ? 1 : 0;
    if (pixelsCount) return pixelsCount; else *right -= 1;
    if ((*top)-- > topBound)
    for (int offset = *left; offset <= *right; ++offset) pixelsCount += image->imageMatrix[*top][offset] ? 1 : 0;
    if (pixelsCount) return pixelsCount; else *top += 1;
    
    if (*bottom < bottomBound && *left > leftBound && image->imageMatrix[*bottom + 1][*left - 1]) {

        *bottom += 1;
        *left -= 1;
        return 1;

    } else if (*bottom < bottomBound && *right < rightBound && image->imageMatrix[*bottom + 1][*right + 1]) {

        *bottom += 1;
        *right += 1;
        return 1;

    } else if (*left > leftBound && *top > topBound && image->imageMatrix[*top - 1][*left - 1]) {

        *left -= 1;
        *top -= 1;
        return 1;

    } else if (*right < rightBound && *top > topBound && image->imageMatrix[*top - 1][*right + 1]) {

        *right += 1;
        *top -= 1;
        return 1;

    } else return 0;

}

static inline int nextRandomValue(int * seed) {

    *seed = *seed * 214013 + 2531011;
    return (*seed >> 16) & 0x7FFF;

}

static inline int removeIndividualPixels(struct Image * restrict image) {

    int pixelsEliminated = 0;
    int bottomBound = image->imageFrame.rowsOffset + image->imageFrame.rowsCount - 1;
    int leftBound = image->imageFrame.columnsOffset;
    int rightBound = image->imageFrame.columnsOffset + image->imageFrame.columnsCount - 1;
    int topBound = image->imageFrame.rowsOffset;

    #pragma omp for
    for (int i = 0; i < PIXEL_REMOVAL_ITERATIONS; ++i) {

        int pixelsCount = -1;
        int seed = rand();
        int column = 0, row = 0;

        do {

            column = image->imageFrame.columnsOffset + nextRandomValue(&seed) % image->imageFrame.columnsCount;
            row = image->imageFrame.rowsOffset + nextRandomValue(&seed) % image->imageFrame.rowsCount;

        } while (!image->imageMatrix[row][column]);

        int bottom = row + PIXEL_BOUNDING_RADIUS;
        bottom = bottom > bottomBound ? bottomBound : bottom;
        int left = column - PIXEL_BOUNDING_RADIUS;
        left = left < leftBound ? leftBound : left;
        int right = column + PIXEL_BOUNDING_RADIUS;
        right = right > rightBound ? rightBound : right;
        int top = row - PIXEL_BOUNDING_RADIUS;
        top = top < topBound ? topBound : top;
        for (int rowOffset = top; rowOffset <= bottom; ++rowOffset)
        for (int columnOffset = left; columnOffset <= right; ++columnOffset) pixelsCount += image->imageMatrix[rowOffset][columnOffset] ? 1 : 0;
        
        if (pixelsCount / ((bottom - top + 1.0) * (right - left + 1.0)) <= PIXEL_REMOVAL_RATIO_VALUE) {

            image->imageMatrix[row][column] = 0;

            #pragma omp atomic
            pixelsEliminated += 1;

    }   }

    return pixelsEliminated;

}

static inline int removeSmallBoundingBoxes(struct Image * restrict image) {

    int pixelsEliminated = 0;
    int bottomBound = image->imageFrame.rowsOffset + image->imageFrame.rowsCount - 1;
    int leftBound = image->imageFrame.columnsOffset;
    int rightBound = image->imageFrame.columnsOffset + image->imageFrame.columnsCount - 1;
    int topBound = image->imageFrame.rowsOffset;

    for (int i = 0; i < BOUNDING_BOX_REMOVAL_ITERATIONS; ++i) {

        int totalPixelsCount = 1;
        int column = 0, row = 0;

        do {

            column = image->imageFrame.columnsOffset + rand() % image->imageFrame.columnsCount;
            row = image->imageFrame.rowsOffset + rand() % image->imageFrame.rowsCount;

        } while (!image->imageMatrix[row][column]);

        int bottom = row, left = column, right = column, top = row;

        do {

            int pixelsCount = growBoundingBox(image, &bottom, bottomBound, &left, leftBound, &right, rightBound, &top, topBound);
            if (pixelsCount) totalPixelsCount += pixelsCount; else break;

        } while ((bottom - top + 1) * (right - left + 1) <= MAXIMUM_BOUNDING_BOX_AREA);

        if ((bottom - top + 1) * (right - left + 1) <= MINIMUM_BOUNDING_BOX_AREA) continue;
        int temporaryBottom = bottom + MINIMUM_VOID_BOUNDING_ZONE;
        temporaryBottom = temporaryBottom > bottomBound ? bottomBound : temporaryBottom;
        int temporaryLeft = left - MINIMUM_VOID_BOUNDING_ZONE;
        temporaryLeft = temporaryLeft < leftBound ? leftBound : temporaryLeft;
        int temporaryRight = right + MINIMUM_VOID_BOUNDING_ZONE;
        temporaryRight = temporaryRight > rightBound ? rightBound : temporaryRight;
        int temporaryTop = top - MINIMUM_VOID_BOUNDING_ZONE;
        temporaryTop = temporaryTop < topBound ? topBound : temporaryTop;
        int pixelsCount = 0;
        for (int x = temporaryTop; x <= temporaryBottom; ++x)
        for (int y = temporaryLeft; y <= temporaryRight; ++y) pixelsCount += image->imageMatrix[x][y] ? 1 : 0;
        if ((double) pixelsCount / totalPixelsCount > PIXELS_COUNT_DIFFERENCE_RATIO_VALUE) continue;
        for (int x = top; x <= bottom; ++x)
        for (int y = left; y <= right; ++y) image->imageMatrix[x][y] = 0;
        pixelsEliminated += totalPixelsCount;

    }

    return pixelsEliminated;

}

static inline int setMainSubFrameBoundingBox(struct Image * restrict image) {

    struct Frame currentFrame;
    currentFrame.columnsCount = 0;
    currentFrame.columnsOffset = 0;
    currentFrame.rowsCount = 0;
    currentFrame.rowsOffset = 0;
    int bottomBound = image->imageFrame.rowsOffset + image->imageFrame.rowsCount - 1;
    int leftBound = image->imageFrame.columnsOffset;
    int rightBound = image->imageFrame.columnsOffset + image->imageFrame.columnsCount - 1;
    int topBound = image->imageFrame.rowsOffset;

    #pragma omp parallel for
    for (int i = 0; i < FRAME_IDENTIFICATION_ITERATIONS; ++i) {

        int seed = rand();
        int column = 0, row = 0;
        struct Frame newFrame;

        do {

            column = image->imageFrame.columnsOffset + nextRandomValue(&seed) % image->imageFrame.columnsCount;
            row = image->imageFrame.rowsOffset + nextRandomValue(&seed) % image->imageFrame.rowsCount;

        } while (!image->imageMatrix[row][column]);

        int bottom = row, left = column, right = column, top = row;
        do if (!growBoundingBox(image, &bottom, bottomBound, &left, leftBound, &right, rightBound, &top, topBound)) break; while (1);
        newFrame.columnsCount = right - left + 1;
        newFrame.columnsOffset = left;
        newFrame.rowsCount = bottom - top + 1;
        newFrame.rowsOffset = top;
        if (newFrame.columnsCount == image->imageFrame.columnsCount &&
            newFrame.columnsOffset == image->imageFrame.columnsOffset &&
            newFrame.rowsCount == image->imageFrame.rowsCount &&
            newFrame.rowsOffset == image->imageFrame.rowsOffset) return 0;
        #pragma omp critical
        if (getFrameArea(&newFrame) > getFrameArea(&currentFrame)) currentFrame = newFrame;

    }

    printf("%d %d %d %d\n", currentFrame.rowsOffset, currentFrame.rowsCount, currentFrame.columnsOffset, currentFrame.columnsCount);

    int finalPixelsCount = 0;
    int initialPixelsCount = 0;
    for (int row = topBound; row <= bottomBound; ++row)
    for (int column = leftBound; column <= rightBound; ++column) initialPixelsCount += image->imageMatrix[row][column] ? 1 : 0;
    image->imageFrame = currentFrame;
    bottomBound = image->imageFrame.rowsOffset + image->imageFrame.rowsCount - 1;
    leftBound = image->imageFrame.columnsOffset;
    rightBound = image->imageFrame.columnsOffset + image->imageFrame.columnsCount - 1;
    topBound = image->imageFrame.rowsOffset;
    for (int row = topBound; row <= bottomBound; ++row)
    for (int column = leftBound; column <= rightBound; ++column) finalPixelsCount += image->imageMatrix[row][column] ? 1 : 0;
    return initialPixelsCount - finalPixelsCount;

}

static inline int shrinkMainFrameBoundingBox(struct Image * restrict image) {

    int pixelsEliminated = 0;

    do {

        int pixelsEliminatedSoFar = pixelsEliminated;
        pixelsEliminated += shrinkMainFrameFromBottom(image);
        pixelsEliminated += shrinkMainFrameFromLeft(image);
        pixelsEliminated += shrinkMainFrameFromRight(image);
        pixelsEliminated += shrinkMainFrameFromTop(image);
        if (pixelsEliminatedSoFar == pixelsEliminated) break;

    } while (1);

    return pixelsEliminated;

}

static inline int shrinkMainFrameFromBottom(struct Image * restrict image) {

    int row = image->imageFrame.rowsOffset + image->imageFrame.rowsCount - 1;
    if (row < image->imageFrame.rowsOffset) return 0;
    int pixelsCount = 0;
    for (int i = 0; i < image->imageFrame.columnsCount; ++i)
    pixelsCount += image->imageMatrix[row][i + image->imageFrame.columnsOffset] ? 1 : 0;

    if ((double) pixelsCount / image->imageFrame.columnsCount <= FRAME_SHRINK_RATIO_VALUE) {

        image->imageFrame.rowsCount -= 1;
        return pixelsCount;

    } else return 0;
    
}

static inline int shrinkMainFrameFromLeft(struct Image * restrict image) {

    int column = image->imageFrame.columnsOffset;
    if (!image->imageFrame.columnsCount) return 0;
    int pixelsCount = 0;
    for (int i = 0; i < image->imageFrame.rowsCount; ++i)
    pixelsCount += image->imageMatrix[i + image->imageFrame.rowsOffset][column] ? 1 : 0;

    if ((double) pixelsCount / image->imageFrame.rowsCount <= FRAME_SHRINK_RATIO_VALUE) {

        image->imageFrame.columnsOffset += 1;
        image->imageFrame.columnsCount -= 1;
        return pixelsCount;

    } else return 0;
    
}

static inline int shrinkMainFrameFromRight(struct Image * restrict image) {

    int column = image->imageFrame.columnsOffset + image->imageFrame.columnsCount - 1;
    if (column < image->imageFrame.columnsOffset) return 0;
    int pixelsCount = 0;
    for (int i = 0; i < image->imageFrame.rowsCount; ++i)
    pixelsCount += image->imageMatrix[i + image->imageFrame.rowsOffset][column] ? 1 : 0;

    if ((double) pixelsCount / image->imageFrame.rowsCount <= FRAME_SHRINK_RATIO_VALUE) {

        image->imageFrame.columnsCount -= 1;
        return pixelsCount;

    } else return 0;
    
}

static inline int shrinkMainFrameFromTop(struct Image * restrict image) {

    int row = image->imageFrame.rowsOffset;
    if (!image->imageFrame.rowsCount) return 0;
    int pixelsCount = 0;
    for (int i = 0; i < image->imageFrame.columnsCount; ++i)
    pixelsCount += image->imageMatrix[row][i + image->imageFrame.columnsOffset] ? 1 : 0;

    if ((double) pixelsCount / image->imageFrame.columnsCount <= FRAME_SHRINK_RATIO_VALUE) {

        image->imageFrame.rowsOffset += 1;
        image->imageFrame.rowsCount -= 1;
        return pixelsCount;

    } else return 0;
    
}

static inline void printImageMatrix(struct Image * restrict image) {

    int columnsBound = image->imageFrame.columnsOffset + image->imageFrame.columnsCount;
    int rowsBound = image->imageFrame.rowsOffset + image->imageFrame.rowsCount;
    
    for (int row = image->imageFrame.rowsOffset; row < rowsBound; ++row) {

        for (int column = image->imageFrame.columnsOffset; column < columnsBound; ++column)
        printf("%c", image->imageMatrix[row][column] ? '#' : ' ');
        printf("\n");

}   }

static inline void refitImageMatrix(struct Image * restrict image) {

    size_t itemSize = sizeof(unsigned char);
    for (int i = 0; i < image->imageFrame.rowsCount; ++i)
    memmove(image->imageMatrix[i],
            image->imageMatrix[i + image->imageFrame.rowsOffset] +
            image->imageFrame.columnsOffset * itemSize,
            image->imageFrame.columnsCount * itemSize);
    for (int i = 0; i < image->imageFrame.rowsCount; ++i)
    memset(image->imageMatrix[i] + image->imageFrame.columnsCount * itemSize, 0,
          (image->columnsCount - image->imageFrame.columnsCount) * itemSize);
    for (int i = image->imageFrame.rowsCount; i < image->rowsCount; ++i)
    memset(image->imageMatrix[i], 0, image->columnsCount * itemSize);
    image->imageFrame.columnsCount = image->columnsCount;
    image->imageFrame.columnsOffset = 0;
    image->imageFrame.rowsCount = image->rowsCount;
    image->imageFrame.rowsOffset = 0;

}
