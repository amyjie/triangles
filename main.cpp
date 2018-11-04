#include "main.h"

/* Wrap around CUDA RT-API calls to automatically catch and print errors */
inline
cudaError_t CUDA_EC(cudaError_t result)
{
#if defined(DEBUG)        
  if(result != cudaSuccess) {
    fprint(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    exit(1);
  }  
#endif  
  return result;
}

int main(int argc, char ** argv)
{	
	/* Turns command line arguments into global parameters. */
	parseArgs(argc, argv);

	/* Open the source image. This is a buffer of pixels in RGBA order. */
  unsigned width, height;
	uint8_t * image = openImage(IMAGE_PATH, width, height);
  std::cout << "Opened: " << IMAGE_PATH << "\t(" << width << "x" << height << ")" << std::endl;

  /* Copy it to the GPU */
  size_t image_size = 4 * width * height;
  uint8_t * cuda_image = 0;
  cudaMallocManaged(&cuda_image, image_size);
  memcpy(cuda_image, image, image_size);

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

  /* Allocate genomes on the GPU */
  size_t genome_size = GENOME_LENGTH * TRIANGLE_SIZE + BG_COLOR_SIZE; 
  size_t max_num_artists = POPULATION_SIZE + NUM_CHILDREN;
  uint8_t * genomes;

  std::vector<Artist> artists(max_num_artists);
  for(Artist & a : artists)
  {
    CUDA_EC(cudaMallocManaged(&(a.genome), genome_size));
    CUDA_EC(cudaMallocManaged(&(a.canvas), image_size));
  }

  /* Create a random byte generator */
  std::mt19937_64 rand_engine(RANDOM_SEED);
  std::independent_bits_engine<std::mt19937_64, 8, uint8_t> rand_byte_generator(rand_engine);

  /* Randomize the genomes, blank out the canvases */
  for(Artist & a : artists)
  {
    for(size_t i = 0; i < image_size; i++)
    {
      if(i < genome_size) {     
        a.genome[i] = rand_byte_generator();         
      }

      a.canvas[i] = 255;
    }
  }
                    





  /* Write the image to website so I can view it */
  char artist_name[] = "artist.png";
  saveImage(artist_name, image, width, height);

  return 0;
}
