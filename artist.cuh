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

struct Artist {
  uint8_t * genome = 0;
  uint8_t * canvas = 0;
  double fitness = 0;
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
void drawTriangle(Pixel * canvas, Triangle_d tri, RGBA color, size_t canvas_size, unsigned width, unsigned height);

#endif
