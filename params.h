#ifndef PARAMS_H 
#define PARAMS_H 

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include <string>
#include <algorithm>

typedef u_int8_t uint8_t;

/* Path to the image that the artist is to draw */
extern char * IMAGE_PATH;

/* Maximum number of triangles to attempt to draw the image with. Defaults to 
   10 triangles.
 */
extern size_t GENOME_LENGTH;

/* Number of artists in each generation. Defaults to 25 */
extern size_t POPULATION_SIZE;

/* Number of children the population produces each generation.
   Defaults to 10.
*/
extern size_t NUM_CHILDREN;

/* The amount of effort put into finding a solution. Effort is equal to:
   POPULATION_SIZE + NUMBER_OF_KIDS * NUM_GENERATIONS.
   Defaults to: 10,000
*/
extern size_t EFFORT;

/* srand() seed for repeatable testing. Defaults to 0. */
extern unsigned int RANDOM_SEED;

/* Chance for a diploid chromosome to crossover. Defaults to 0.7. */
extern double XOVER_CHANCE;

/* Sets how crossover proceeds.
   - "bit"      -> the genome will crossover at an individual bit boundaries
   - "byte"     -> the genome will crossover at byte boundaries
   - "triangle" -> the genome will crossover at triangle boundaries
 */
enum class Xover_type {BIT, BYTE, TRIANGLE};
extern Xover_type XOVER_TYPE;

/* Chance, per bit, of being flipped each generation. Defaults to 0.005. */ 
extern double MUTATION_RATE;

/* Chance to swap two triangles, or randomize a triangle. Defaults to 0.5 */
extern double MACRO_MUTATION_RATE;

/* Puts all the arguments into the variables */
void parseArgs(int argc, char ** argv);

#endif
