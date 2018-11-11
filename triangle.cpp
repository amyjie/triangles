#include "triangle.h"

/* Mark important _byte_ offsets */
size_t BG_COLOR_OFFSET = 0;
size_t BG_COLOR_SIZE = 4;
size_t TRIANGLE_LIST_BEGIN = BG_COLOR_SIZE;
size_t TRIANGLE_SIZE = sizeof(Triangle);
size_t VIS_BYTE_INDEX = TRIANGLE_SIZE - 1;

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

/* Returns a random triangle */
Triangle genRandTriangle(std::independent_bits_engine<std::mt19937_64, 8, uint8_t> rand_byte_generator)
{
  Triangle tri;

  tri.x1 = rand_byte_generator();
  tri.y1 = rand_byte_generator();

  tri.x2 = rand_byte_generator();
  tri.y2 = rand_byte_generator();

  tri.x3 = rand_byte_generator();
  tri.y3 = rand_byte_generator();

  tri.r = rand_byte_generator();
  tri.g = rand_byte_generator();
  tri.b = rand_byte_generator();
  tri.a = rand_byte_generator();

  tri.visible = rand_byte_generator();

  return tri;

}
