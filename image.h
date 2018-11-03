#ifndef IMAGE_H 
#define IMAGE_H

#include <iostream>

#include <stdio.h>
#include <stdlib.h>

#include "lodepng.h"

#define MAX_PNG_HEIGHT 1024
#define MAX_PNG_WIDTH 1024

typedef u_int8_t uint8_t;

typedef unsigned int image_error_t;

/* Opens the file given by filename, and returns a unint8_t buffer, and sets
   the value of width and height to the width and height of the image.

   Since opening an image is critical to this program, if it encounters an error
   it will print the error and exit the program.

   The buffer is 4 * width * height bytes in size and the caller is repsonsible
   for freeing it.
*/
uint8_t * openImage(char * filename, unsigned & width, unsigned & height);

#endif
