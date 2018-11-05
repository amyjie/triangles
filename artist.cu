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
inline
bool edgeFunction(float x1, float y1, float x2, float y2, float px3, float py3)
{
  return ((px3 - x1) * (y2 - y1) - (py3 - y1) * (x2 - x1) >= 0);
}

/* Convenience function for compositing two colors */
__device__
inline
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
//void drawTriangle(Pixel * canvas, Triangle_d tri, RGBA color, size_t canvas_size, unsigned width, unsigned height)
void drawTriangle(Pixel * canvas, Triangle_d tri, RGBA color, size_t canvas_size, unsigned width, unsigned height, float max_x, float min_x, float max_y, float min_y)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= canvas_size) { return; }

  float px = (float)(index % width);
  float py = (float)(index / width);

  if(px > max_x || px < min_x || py > max_y || py < min_y) { return; }

  bool draw_pixel = true;
  draw_pixel &= edgeFunction(tri.x1, tri.y1, tri.x2, tri.y2, px, py);
  draw_pixel &= edgeFunction(tri.x2, tri.y2, tri.x3, tri.y3, px, py);
  draw_pixel &= edgeFunction(tri.x3, tri.y3, tri.x1, tri.y1, px, py);

  if(draw_pixel) {
    canvas[index] = blendColors(canvas[index], color);
  }
}

/* Returns the average, per-pixel, per-channel RMSE */
__global__
void gradeArt(uint8_t * canvas, uint8_t * image, size_t image_size, double * diff)
{

  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < image_size) {

    double sq_error = (canvas[index] - image[index]) * (canvas[index] - image[index]);
    double avg_error = sq_error / (double)image_size;

    diff[index] = avg_error;
  }
}

/* A metaphor for life. Takes an artists error grid and computes partial sums,
   accumulating them into a.error so they can be summed in the CPU becaues I 
   don't have time to learn how to do recursive, efficient, parallel summation.
*/
__global__
void accumulateErrors(double * input, double * output, size_t len)
{
  /* Load part of the input into shared memory to improve partial sum */
  __shared__ double partial_sum[2 * GABS];

  size_t index = threadIdx.x;
  size_t start = 2 * GABS * blockIdx.x;

  if(start + index < len) {
    partial_sum[index] = input[start + index];
  } else {
    partial_sum[index] = 0;
  }

  if(start + GABS + index < len) {
    partial_sum[GABS + index] = input[start + GABS + index];
  } else {
    partial_sum[GABS + index] = 0;
  }
 
  /* Begin reducing the sums in shared memory */
  for(size_t stride = GABS; stride >= 1; stride >>= 1) {
    __syncthreads();
    if(index < stride) {
      partial_sum[index] += partial_sum[index + stride];
    }
  }

  /* Write the partial sum back out */
  if(index == 0) {
    output[blockIdx.x] = partial_sum[0];
  }
}
