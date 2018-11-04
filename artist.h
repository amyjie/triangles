#ifndef ARTIST_H
#define ARTIST_H

#include "triangle.h"

/* Helpful for the bit twiddling */
#define GETMASK(index, size) (((1 << (size)) - 1) << (index))

/* Takes a uint8_t and turns it into a double [0,1) */
#define UITD(num, dimension) (((double)num / 255) * dimension)

struct Artist {
  uint8_t * genome = 0;
  uint8_t * canvas = 0;
  double fitness = 0;
};

#endif
