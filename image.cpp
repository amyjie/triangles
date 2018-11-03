#include "image.h"

/* Take the name of a PNG and returns a vector of ints in RGB order */
uint8_t * openImage(char * filename, unsigned & width, unsigned & height)
{
  
  uint8_t * image;
  image_error_t error;
  error = lodepng_decode32_file(&image, &width, &height, filename); 

  if(error) {
    printf("Error #%u in opening file: %s", error, lodepng_error_text(error));
  }

  return image;
}
