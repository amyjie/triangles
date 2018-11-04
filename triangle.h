#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <cstdlib>

/* Takes a uint8_t and turns it into a float [0,1) */
#define UITF(num) (((float)num / 255))

typedef u_int8_t uint8_t;

/*
[xxxxxxxxxx][yyyyyyyyyy][xxxxxxxxxx][yyyyyyyyyy][xxxxxxxxxx][yyyyyyyyyy]|[rrrrrrrr][gggggggg][bbbbbbbb]|[aaaaaaa]|[v]
[                                points                                ]|[           colors           ]|[opacity ]|[visible ]

Total size: 11 bytes.
Packed to preserve alignment.
*/

struct __attribute__((__packed__)) Triangle {
  /* First Point */
  uint8_t x1 : 8;
  uint8_t y1 : 8;
  
  /* Second Point */
  uint8_t x2 : 8;
  uint8_t y2 : 8;

  /* Third Point */
  uint8_t x3 : 8;
  uint8_t y3 : 8;

  /* Color */
  uint8_t r : 8;
  uint8_t g : 8;
  uint8_t b : 8;
  uint8_t a : 7;

  /* If visible */
  bool visible : 1;
};

/* 4 bytes */
struct Pixel {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
};

typedef Pixel RGBA;

/* Can't send bit packed structures to GPUs */
struct Triangle_d {
  float x1;
  float y1;  
  float x2;
  float y2;
  float x3;
  float y3;
};

/* Copies a Host Triangle to a Device Triangle_d */

/* Mark important offsets */
extern size_t BG_COLOR_OFFSET;
extern size_t BG_COLOR_SIZE;
extern size_t TRIANGLE_LIST_BEGIN;
extern size_t TRIANGLE_SIZE;

/* Copies a Host Triangle to a Device Triangle_d */
Triangle_d convertTriangleH2D(Triangle & tri, unsigned width, unsigned height);

/* Copies the RGBA information of a Triangle to a RGBA struct */
RGBA convertRGBA(Triangle & tri);

#endif
