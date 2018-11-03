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

  /* Copy the image to the website */
  std::ifstream src(filename, std::ios::binary);
  std::ofstream dst("/var/www/html/tri/original.png", std::ios::binary);
  dst << src.rdbuf();
  src.close();
  dst.close();

  return image;
}

/* Writes a buffer to a PNG at the current location. Does not free the buffer.
*/
void saveImage(char * filename, uint8_t * img, unsigned width, unsigned height)
{
  image_error_t error = lodepng_encode32_file(filename, img, width, height);
  if(error) {
    printf("Error #%u: %s", error, lodepng_error_text(error));
  }

  /* Copy the image to the website */
  std::ifstream src(filename, std::ios::binary);
  std::ofstream dst("/var/www/html/tri/artist.png", std::ios::binary);
  dst << src.rdbuf();
  src.close();
  dst.close();
}

