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
  char artist_name[] = "artist.png";
//  std::cout << "Opened: " << IMAGE_PATH << "\t(" << width << "x" << height << ")" << std::endl;

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
  size_t num_bits    = genome_size * 8;
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
  std::exponential_distribution<double> artist_repro_dist(3.5);
  std::binomial_distribution<size_t> b_dist(num_bits, MUTATION_RATE);
  std::uniform_int_distribution<size_t> bit_picker(0, (num_bits - 1));

  /* Create a CUDA stream, randomize the genomes */
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
  }

  /* Main loop */
  size_t effort = POPULATION_SIZE;
  double best_fitness = 0;
  for(;effort < EFFORT;)
  { 
    /* Draw the canvas background colors */
    for(Artist & a : artists)
    {
      /* If the triangle already has a fitness, it remains unchanged. */      
      if(a.fitness != 0) { continue; }

      /* The first bytes of a genome is the background RGBA value */
      RGBA bg_color = ((RGBA *)(a.genome))[0];
      /* Cast the canvas into a series of pixels (instead of bytes) */
      RGBA * canvas = (RGBA *)a.canvas;
      setCanvasColor<<<BAT(image_num_pixels,256),0,a.stream>>>(canvas, image_num_pixels, bg_color);
    }

    /* Draw the triangles to their respective canvases */
    for(Artist a : artists)
    {
      /* If the triangle already has a fitness, it remains unchanged. */      
      if(a.fitness != 0) { continue; }

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

        drawTriangle<<<BAT(image_num_pixels,128),0,a.stream>>>((Pixel *)a.canvas, tri_d, color, image_num_pixels, width, height, max_x, min_x, max_y, min_y);
      }
    }

    /* Grade each canvas */
    for(Artist a : artists)
    {
      /* If the triangle already has a fitness, it remains unchanged. */      
      if(a.fitness != 0) { continue; }

      gradeArt<<<BAT(image_size,512),0,a.stream>>>(a.canvas, image_d, image_size, a.diff);
    }  

    /* Accumulate errors into a sum */
    for(Artist a : artists)
    {
      /* If the triangle already has a fitness, it remains unchanged. */      
      if(a.fitness != 0) { continue; }

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

    /* Sort the artists by descending fitness */
    std::sort(artists.begin(), artists.end(), [](Artist & a, Artist & b) {
      return a.fitness > b.fitness;    
    });

    /* Calculate the number of children an artist should have */
    double total_reproduce = 0;
    for(Artist & a : artists)
    {
      if(std_dev == 0) { 
        a.reproduce = 1;
      } else {
        a.reproduce = 1 + ((a.fitness - avg_fitness) / (2 * std_dev));
        if(a.reproduce < 0) { a.reproduce = 0.1; }
      }
      total_reproduce += a.reproduce;
    }

    /* Replace reproduce with a percent chance to reproduce [0,1) */ 
    for(Artist & a : artists)
    {
      a.reproduce /= total_reproduce;
    }

    /* Generate the new generation, overwriting the artists below 
       POPULATION_SIZE. Artists with higher fitness are more likely to
       reproduce.
    */
    size_t kid_index = POPULATION_SIZE;
    size_t sub_index = 0;
    for(size_t i = 0; i < NUM_CHILDREN; i++)
    {
      double r = 2;
      while(r > 1.0) {
        r = artist_repro_dist(rand_engine);
      }

      Artist mate_p;
      Artist mate_s;
      for(Artist a : artists) {
        if(a.reproduce > r) {
          mate_p = a;
          break;
        } else {
          r -= a.reproduce;
        } 
      }

      mate_s = artists[sub_index++];
      sub_index %= POPULATION_SIZE;
      
      /* Baby needs to keep the pointes to the on device memory from the Artist
         it is replacing. It will have it's genome copied over by its parents and
         fitness set to 0 to signal that it needs to be redrawn.
      */   
      Artist & baby = artists[kid_index++];
      baby.fitness = 0;

      /* Only crossover $(crossover_chance)% of the time */
      r = (double)(rand()/RAND_MAX);
      if(r < XOVER_CHANCE)
      {
        /* Where in the genome to crossover at */
        size_t byte_index = 0;
        
        /* Choose a random byte to begin crossover in the genome. If XOVER_TYPE 
           is set to TRIANGLE - make sure that byte is aligned to the start of a
           Triangle struct.
         */
        if(XOVER_TYPE == Xover_type::TRIANGLE)
        {
          /* Generate a random, Triangle aligned, byte index */
          double div = (RAND_MAX/(GENOME_LENGTH));
          size_t tri_index = (size_t)(std::floor((((double)rand() - 1)/div)));
          byte_index = TRIANGLE_LIST_BEGIN + (tri_index * sizeof(Triangle));
        }
        else
        {
          /* Generate a random byte index */
          double div = (RAND_MAX/(genome_size));
          byte_index = (size_t)(std::floor((((double)rand() - 1)/div)));
        }

        /* Swap the bytes in the range: [(byte_index + 1), genome_length].      
           There are no bytes to copy when crossover happens in the last byte.
         */
        if(byte_index <= (genome_size - 1))
        {
          /* Number of bytes to swap */
          memcpy(baby.genome, mate_p.genome, byte_index);
          memcpy((baby.genome + byte_index), (mate_s.genome + byte_index), (genome_size - byte_index));
        }
      }

      /* Mutate the genome */
      size_t num_mutations = b_dist(rand_engine);
      for(size_t i = 0; i < num_mutations; i++)
      {
        /* Indices into bits & bytes */
        size_t bit_index = bit_picker(rand_engine);
        size_t byte_index = bit_index / 8;
        uint8_t intra_bit_index = bit_index % 8;
        
        uint8_t mask = (uint8_t)GETMASK(intra_bit_index, 1);  
        baby.genome[byte_index] ^= mask;
      }
    }

    /* Update the effort */
    effort += NUM_CHILDREN;

    /* Save the image to the web :3 */
    if(best_fitness != artists[0].fitness) { 
      best_fitness = artists[0].fitness;
      saveImage(artist_name, artists[0].canvas, width, height);
      std::cout << effort << "\t" << best_fitness << "\t" << avg_fitness << "\t" << std_dev << "\n";
    }

    /* Print things */
    //output_file << effort << "\t" << best_fitness << "\t" << avg_fitness << "\t" << std_dev << "\n";
  }     

  output_file.close();

  return 0;
}
