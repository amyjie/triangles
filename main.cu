#include "main.cuh"

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
  size_t image_num_pixels = width * height;
  size_t image_size = 4 * image_num_pixels;
  size_t num_error_blocks = image_size / GABS;
  uint8_t * image_d = 0;
  cudaMallocManaged(&image_d, image_size);
  memcpy(image_d, image, image_size);

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

  /* Allocate genomes, and canvases on the GPU */
  size_t genome_size = GENOME_LENGTH * TRIANGLE_SIZE + BG_COLOR_SIZE; 
  size_t max_num_artists = POPULATION_SIZE + NUM_CHILDREN;

  std::vector<Artist> artists(max_num_artists);
  for(Artist & a : artists)
  {
    CUDA_EC(cudaMallocManaged(&(a.genome), genome_size));
    CUDA_EC(cudaMallocManaged(&(a.canvas), image_size));
    CUDA_EC(cudaMallocManaged(&(a.diff),   image_size * sizeof(double)));
    CUDA_EC(cudaMallocManaged(&(a.error),  (image_size / GABS) * sizeof(double)));
  }

  /* Create a random byte generator */
  std::mt19937_64 rand_engine(RANDOM_SEED);
  std::independent_bits_engine<std::mt19937_64, 8, uint8_t> rand_byte_generator(rand_engine);

  /* Create a CUDA stream, randomize the genomes, blank out the canvases */
  for(Artist & a : artists)
  {
    /* Create the streams for each artist */
    CUDA_EC(cudaStreamCreate(&(a.stream)));

    for(size_t i = 0; i < image_size; i++)
    {
      if(i < genome_size) {     
        a.genome[i] = rand_byte_generator();         
      }
    }
    /* The first bytes of a genome is the background RGBA value */
    RGBA bg_color = ((RGBA *)(a.genome))[0];
    /* Cast the canvas into a series of pixels (instead of bytes) */
    RGBA * canvas = (RGBA *)a.canvas;
    setCanvasColor<<<BAT(image_num_pixels,256)>>>(canvas, image_num_pixels, bg_color);
  }

  /* Draw the triangles to their respective canvases */
  for(Artist a : artists)
  {
    Triangle * triangles = (Triangle *)(a.genome + TRIANGLE_LIST_BEGIN);
    for(size_t i = 0; i < GENOME_LENGTH; i++)
    {
      Triangle tri = triangles[i];
      if(tri.visible == 0) { continue; }

      /* Convert the Triangle into unpacked structs for the device. */      
      Triangle_d tri_d = convertTriangleH2D(tri, width, height);
      RGBA color = convertRGBA(tri);
      
      float max_x = MAX(MAX(tri_d.x1, tri_d.x2), tri_d.x3);
      float max_y = MAX(MAX(tri_d.y1, tri_d.y2), tri_d.y3);

      float min_x = MIN(MIN(tri_d.x1, tri_d.x2), tri_d.x3);
      float min_y = MIN(MIN(tri_d.y1, tri_d.y2), tri_d.y3);

      drawTriangle<<<BAT(image_num_pixels,128)>>>((Pixel *)a.canvas, tri_d, color, image_num_pixels, width, height, max_x, min_x, max_y, min_y);
    }
  }
  cudaDeviceSynchronize();
 // std::cout << artists[0].fitness << std::endl;

  /* Grade each canvas */
  for(Artist a : artists)
  {
    gradeArt<<<BAT(image_size,512)>>>(a.canvas, image_d, image_size, a.diff);
  }  
  cudaDeviceSynchronize();

  /* Accumulate errors into a sum */
  for(Artist a : artists)
  {
    accumulateErrors<<<BAT(image_size,GABS),GABS,a.stream>>>(a.diff, a.error, image_size);
  }  
  cudaDeviceSynchronize();

  for(Artist & a : artists)
  {
    double error = 0;
    for(size_t i = 0; i < num_error_blocks; i++)
    {
      error += a.error[i]; 
    }
    error /= num_error_blocks;
    a.fitness = 1 / std::sqrt(error);
  }

	/* Calculate the average fitness & std dev of the artists */
	double std_dev = 0.0;
	double avg_fitness = 0.0;
	for(Artist a : artists) 
	{
		avg_fitness += a.fitness;
	}
	avg_fitness /= max_num_artists;

	/* Calculate the std dev of the artists */
	for(Artist a : artists) 
	{
		std_dev += pow(a.fitness - avg_fitness, 2);
		std_dev = sqrt(std_dev/max_num_artists);
	}

  std::cout << "Avg Fitness: " << avg_fitness << std::endl;
  std::cout << "Std Dev: " << std_dev << std::endl;

  /* Sort the artists by descending fitness */
  std::sort(artists.begin(), artists.end(), [](Artist & a, Artist & b) {
    return a.fitness > b.fitness;    
  });



//  std::cout << artists[0].fitness << std::endl;

                    





  /* Write the image to website so I can view it */
  char artist_name[] = "artist.png";
  //saveImage(artist_name, image, width, height);
  saveImage(artist_name, artists[0].canvas, width, height);


  return 0;
}
