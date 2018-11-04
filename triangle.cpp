#include "triangle.h"

/* Mark important _byte_ offsets */
size_t BG_COLOR_OFFSET = 0;
size_t BG_COLOR_SIZE = 4;
size_t TRIANGLE_LIST_BEGIN = BG_COLOR_SIZE;
size_t TRIANGLE_SIZE = sizeof(Triangle);

/* Copies a Host Triangle to a Device Triangle_d */
Triangle_d convertTriangleH2D(Triangle & tri, unsigned width, unsigned height)
{
  Triangle_d tri_d;
  
  tri_d.x1 = UITF(tri.x1) * width;
  tri_d.y1 = UITF(tri.y1) * height;

  tri_d.x2 = UITF(tri.x2) * width;
  tri_d.y2 = UITF(tri.y2) * height;
 
  tri_d.x3 = UITF(tri.x3) * width;
  tri_d.y3 = UITF(tri.y3) * height;

  return tri_d;
}

/* Copies the RGBA information of a Triangle to a RGBA struct */
RGBA convertRGBA(Triangle & tri)
{
  RGBA color;

  color.r = tri.r;
  color.g = tri.g;
  color.b = tri.b;
  color.a = tri.a;

  return color;
}
