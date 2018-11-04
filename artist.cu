#include "artist.cuh"

/* Scores a canvas */

/* Sets the canvas to all white, opcaity 100% */
__global__
void blankCanvas(uint8_t * canvas, size_t canvas_size)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < canvas_size) {
    canvas[index] = 255;
  }
}

/* Sets the canvas to the specified color */
__global__
void setCanvasColor(RGBA * canvas, size_t canvas_size, RGBA color)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < canvas_size) {
    canvas[index].r = color.r;
    canvas[index].g = color.g;
    canvas[index].b = color.b;
    canvas[index].a = color.a;
  }
}
/* Convenience function for calculating edges */
__device__
bool edgeFunction(float x1, float y1, float x2, float y2, float px3, float py3)
{
  return ((px3 - x1) * (y2 - y1) - (py3 - y1) * (x2 - x1) >= 0);
}

/* Convenience function for compositing two colors */
__device__
Pixel blendColors(RGBA color1, RGBA color2)
{
  Pixel p; 
  
  /* Convert to floats [0,1) */
  float a1 = UITF(color1.a);
  float a2 = UITF(color2.a);
  float a = a1 + (a2 * (1 - a1));

  float r1 = UITF(color1.r) * a1;
  float r2 = UITF(color2.r) * a2;
  float r = r1 + (r2 * (1 - a1));

  float g1 = UITF(color1.g) * a1;
  float g2 = UITF(color2.g) * a2;
  float g = g1 + (g2 * (1 - a1));

  float b1 = UITF(color1.b) * a1;
  float b2 = UITF(color2.b) * a2;
  float b = b1 + (b2 * (1 - a1));

  p.r = r * 255;
  p.g = g * 255;
  p.b = b * 255;
  p.a = a * 255;

  return p;

}

/* Draws a triangle to a canvas */
__global__
void drawTriangle(Pixel * canvas, Triangle_d tri, RGBA color, size_t canvas_size, unsigned width, unsigned height)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= canvas_size) { return; }

  float px = (float)(index % width);
  float py = (float)(index / width);

  bool draw_pixel = true;
  draw_pixel &= edgeFunction(tri.x1, tri.y1, tri.x2, tri.y2, px, py);
  draw_pixel &= edgeFunction(tri.x2, tri.y2, tri.x3, tri.y3, px, py);
  draw_pixel &= edgeFunction(tri.x3, tri.y3, tri.x1, tri.y1, px, py);

  if(draw_pixel) {
    canvas[index] = blendColors(canvas[index], color);
  }
}
