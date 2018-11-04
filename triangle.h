#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <cstdlib>

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

/* Mark important offsets */
extern size_t BG_COLOR_OFFSET;
extern size_t BG_COLOR_SIZE;
extern size_t TRIANGLE_LIST_BEGIN;
extern size_t TRIANGLE_SIZE;

/* Compare two triangles */
bool operator==(const Triangle& lhs, const Triangle& rhs);


#endif
