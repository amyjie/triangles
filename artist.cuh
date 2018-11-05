#ifndef ARTIST_H
#define ARTIST_H

#include <cuda_runtime.h>

#include <stdio.h>

#include "triangle.h"

/* Helpful for the bit twiddling */
#define GETMASK(index, size) (((1 << (size)) - 1) << (index))

/* Takes a uint8_t and turns it into a double [0,1) */
#define UITD(num, dimension) (((double)num / 255) * dimension)

/* Takes a uint8_t and turns it into a float [0,1) */
#define UITF(num) (((float)num / 255))

/* Macro to fill in the specifics of a kernel <<<>>> */
#define BAT(size, threads) ((size + (threads - 1)) / threads), (threads)

/* Macro to return max/min numbers */
#define MAX(x, y) (x >= y ? x : y)
#define MIN(x, y) (x < y ? x : y)

/* Size of Block for gradeArt() */
#define GABS 512

struct Artist {
  /* Background color and list of triangles */      
  uint8_t * genome = 0;
  /* RGBA uint8_t's representing pixels where the artist can draw to. */
  uint8_t * canvas = 0;
  /* Same size as canvas, the MSE's per channel are stored here. */
  double  * diff   = 0;
  /* Size of the canvas / GABS. This is space is used to accumulate errors. */
  double  * error  = 0;
  /* Convert the errors to a fitness score to rank artists */
  double   fitness = 0;
  /* The amount of times this artist is allowed to reproduce */
  double reproduce = 0;
  /* So the artists don't block each other */
  cudaStream_t stream;
};

/* Sets the canvas to all white, opcaity 100% */
__global__
void blankCanvas(uint8_t * canvas, size_t canvas_size);

/* Sets the canvas to the color specified */
__global__
void setCanvasColor(RGBA * canvas, size_t canvas_size, RGBA color);

/* Draws a triangle to a canvas */
__global__
void drawTriangle(Pixel * canvas, Triangle_d tri, RGBA color, size_t canvas_size, unsigned width, unsigned height, float max_x, float min_x, float max_y, float min_y);

/* Returns the average, per-pixel, per-channel RMSE */
__global__
void gradeArt(uint8_t * canvas, uint8_t * image, size_t image_size, double * diff);

/* A metaphor for life. Takes an artists error grid and computes partial sums,
   accumulating them into a.error so they can be summed in the CPU becaues I 
   don't have time to learn how to do recursive, efficient, parallel summation.
*/
__global__
void accumulateErrors(double * input, double * output, size_t len);

#endif
