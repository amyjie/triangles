#include "main.h"

int main(int argc, char ** argv)
{	
	/* Turns command line arguments into global parameters. */
	parseArgs(argc, argv);

	/* Open the source image. This is a buffer of pixels in RGBA order. */
  unsigned width, height;
	uint8_t * image = openImage(IMAGE_PATH, width, height);

  /* Open a file to write the results */
  std::ofstream output_file;

  std::string image_path(IMAGE_PATH);
  std::string genome_length(std::to_string(GENOME_LENGTH));
  std::string pop_size(std::to_string(POPULATION_SIZE));
  std::string seed(std::to_string(RANDOM_SEED));
  std::string xover(std::to_string(XOVER_CHANCE));
  std::string mrate(std::to_string(MUTATION_RATE));

  std::string file_name = image_path;
  file_name += "_" + genome_length;
  file_name += "_" + pop_size;
  file_name += "_" + xover;
  file_name += "_" + mrate;
  file_name += "_" + seed;

  output_file.open("output/" + file_name + ".tsv");

  std::cout << "Opened: " << IMAGE_PATH << "\t(" << width << "x" << height << ")" << std::endl;

  return 0;
}
