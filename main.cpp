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
   
  /* Allocate the genomes in pinned host memory */
  /* Let the driver know that some memory allocations will be pinned */
  CUDA_EC(cudaSetDeviceFlags(cudaHostAllocMapped));

  /* Calculate the size of the buffer */
  size_t genome_size = GENOME_LENGTH * TRIANGLE_SIZE + BG_COLOR_SIZE; 
  size_t population_buffer_size = genome_size * POPULATION_SIZE;

  /* genomes is a point to the buffer of all the genomes in the population */
  void ** genomes;
  unsigned int alloc_flags = cudaHostAllocPortable || cudaHostAllocMapped ||
          cudaHostAllocWriteCombined;

  /* And away we go... */
  cudaHostAlloc(genomes, population_buffer_size, alloc_flags);












  /* Write the image to website so I can view it */
  char artist_name[] = "artist.png";
  saveImage(artist_name, image, width, height);

  return 0;
}
