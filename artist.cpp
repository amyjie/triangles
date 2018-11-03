#include "artist.h"

/* * * * * * * * * * * * * * * * * * *
 *    Static Member Initialization   *
 * * * * * * * * * * * * * * * * * * */

/* Random engine, independent bytes generator */
std::default_random_engine Artist::rand_engine;
std::independent_bits_engine<std::default_random_engine, 8, unsigned char> Artist::rand_byte_generator;

/* Default Artist values */
size_t Artist::number_of_triangles = 10;
size_t Artist::genome_length = (Artist::number_of_triangles * sizeof(Triangle) + 4 * sizeof(uint8_t));
size_t Artist::expression_limit = 0;
size_t Artist::count = 0;

double Artist::mutation_rate = 0.005;
double Artist::crossover_chance = 0.7;

Xover_type Artist::crossover_type = Xover_type::BIT;

Magick::Image Artist::source_copy;
bool Artist::source_img_hash[IMG_HASH_SQRT * IMG_HASH_SQRT];

/* * * * * * * * * * * * * * * * * * *
 *          Helper Functions         *
 * * * * * * * * * * * * * * * * * * */

/* Takes an Image and for each pixel (except the ones in the left most column)
   and sets it as true if it is lighter than the pixel to the left, and false
   otherwise.

   image_hash is an array and should be of length IMG_HASH_SQRT * (IMG_HASH_SQRT + 1)
 */
void calculateImageHash(Magick::Image const & image, bool * image_hash)
{
  size_t length = image.columns() * image.rows();
  size_t img_hash_index = 0;

  double previous_value = 0.0;
  for(size_t i = 0; i < length; i++)
  {
    /* Check if this pixel is lighter (lower grayscale value) than the previous
       pixel (pixel to its left).
     */
    size_t x = i % (IMG_HASH_SQRT + 1);
    size_t y = i / (IMG_HASH_SQRT + 1);
    Magick::Color color = image.pixelColor(x, y);
    
    /* Grayscale as the human eye perceives it */
    double r = 0.21 * color.quantumRed();
    double g = 0.72 * color.quantumGreen();
    double b = 0.07 * color.quantumBlue();
    double value = r + g + b;

    /* If this is a pixel in the "left most" column we don't need to calculate
       the bool.
     */
    if(i % (IMG_HASH_SQRT + 1) != 0)
    {
      image_hash[img_hash_index++] = (value > previous_value);
    }

    previous_value = value;
  }
}

/* Calcuate Hamming Distance between two arrays of equal length */
inline size_t calculateHammingDistance(bool * source_hash, bool * image_hash, size_t length)
{
  size_t distance = 0;
  for(size_t i = 0; i < length; i++)
  {
    if(source_hash[i] != image_hash[i]) { distance++; }
  }
  return distance;
}

/* * * * * * * * * * * * * * * * * * *
 *  Static Initialization Functions  *
 * * * * * * * * * * * * * * * * * * */

/* Calls all the other initialization functions - allows the Artist class to be
   setup in one step.
 */
void Artist::initialization(size_t RANDOM_SEED, size_t GENOME_LENGTH, double MUTATION_RATE, double XOVER_CHANCE, Xover_type XOVER_TYPE)
{
  Artist::initializeRandomByteGenerator(RANDOM_SEED);
  Artist::initializeGenomeLength(GENOME_LENGTH);
  Artist::initializeMutationRate(MUTATION_RATE);
  Artist::initializeCrossoverChance(XOVER_CHANCE);
  Artist::initializeCrossoverType(XOVER_TYPE);
}

/* Set up the static random byte generator */
void Artist::initializeRandomByteGenerator(size_t RANDOM_SEED)
{
  Artist::rand_engine.seed(RANDOM_SEED);
  std::independent_bits_engine<std::default_random_engine, 8, unsigned char> dummy(Artist::rand_engine);
  Artist::rand_byte_generator = dummy;
}

/* Set the max number of triangles, and genome byte length */
void Artist::initializeGenomeLength(size_t GENOME_LENGTH)
{
  /* The CLI parameter "GENOME_LENGTH" actually specifys the number of 
     triangles. This is because the user won't know how many bytes to specify.
   */
  Artist::number_of_triangles = GENOME_LENGTH;
  Artist::expression_limit = GENOME_LENGTH;

  /* Calculate the byte length */
  size_t size_of_bg_rbg = (4 * sizeof(uint8_t));
  size_t size_of_triangles = (Artist::number_of_triangles * sizeof(Triangle));
  Artist::genome_length = size_of_bg_rbg + size_of_triangles;
}

/* Set the mutation rate */
void Artist::initializeMutationRate(double MUTATION_RATE)
{
  if(MUTATION_RATE >= 0 && MUTATION_RATE <= 1) { Artist::mutation_rate = MUTATION_RATE; }
  else 
  { 
    std::cerr << "Mutation rate must be between 0.0 and 1.0.\nRate \
                  provided: " << MUTATION_RATE << std::endl;
    exit(1);
  }
}

/* Set the crossover chance */
void Artist::initializeCrossoverChance(double XOVER_CHANCE)
{
  if(XOVER_CHANCE >= 0 && XOVER_CHANCE <= 1) { Artist::crossover_chance = XOVER_CHANCE; }
  else 
  { 
    std::cerr << "Crossover chance must be between 0.0 and 1.0.\nChance \
                  provided: " << XOVER_CHANCE << std::endl;
    exit(1);
  }
}

/* Set what boundaries are we allowed to crossover at */
void Artist::initializeCrossoverType(Xover_type XOVER_TYPE)
{
  Artist::crossover_type = XOVER_TYPE;
}

/* Stores a copy of the source image, and a miniture version for scoring. */
void Artist::initializeSourceImage(Magick::Image const & source)
{
  /* Copy the source into the new images */
  Artist::source_copy = source;
  Magick::Image small_source_copy = source;

  /* Generate the dimension string */
  size_t width  = (IMG_HASH_SQRT + 1);
  size_t height = IMG_HASH_SQRT;
  std::string dimensions = "!" + std::to_string(width) + "x!" + std::to_string(height);

  /* Resize the image */
  small_source_copy.resize(dimensions);

  /* Calculate the image hash */
  calculateImageHash(small_source_copy, Artist::source_img_hash);
}

/* * * * * * * * * * * * * * * * * * *
 *            Contructors            *
 * * * * * * * * * * * * * * * * * * */


/* Generates an artist with a random genotype, with only the first
   triangle set as visible.
 */
Artist::Artist()
{
  chromosome.dominant = (uint8_t *) malloc(Artist::genome_length);
  chromosome.recessive = (uint8_t *) malloc(Artist::genome_length);

  /* Randomize the bits in the byte array */
  for(size_t i = 0; i < (Artist::genome_length); i++)
  {
    chromosome.dominant[i] = rand_byte_generator(); 
    chromosome.recessive[i] = rand_byte_generator(); 
  }

  /* Set only the first triangle to visible */
  Triangle * dom_tri = (Triangle *)(chromosome.dominant + TRIANGLE_LIST_BEGIN);
  Triangle * rec_tri = (Triangle *)(chromosome.recessive + TRIANGLE_LIST_BEGIN);
  for(size_t i = 0; i < Artist::number_of_triangles; i++)
  {
    dom_tri[i].visible = 0;
    rec_tri[i].visible = 0;

    /* Set the first triangle's visiblity to 00000111 */
    if(i == 0) { dom_tri[i].visible = 7; }
  }

  /* Set the fitness as high as possible to ensure it's replaced. */
  // fitness = std::numeric_limits<double>::max();
  fitness = 0;
  expected_reproduction = 1.0; /* This will be overwritten later */

  /* Sets the location index */
  location_index = Artist::count++;

  /* New artists are not clones */
  is_clone = false;
}

/* Generates a new Artist from two parent artists. Caller is responsible
   for freeing the returned baby.
 */
Artist::Artist(const Artist &a, const Artist &b)
{
  /* Allocate the memory for the chromosome */
  chromosome.dominant = (uint8_t *) malloc(Artist::genome_length);
  chromosome.recessive = (uint8_t *) malloc(Artist::genome_length);

  /* Copy the bits into the new chromosome. A gives the dominant genome, B 
     gives the recessive genome.
   */
  memcpy(chromosome.dominant, a.chromosome.dominant, Artist::genome_length);
  memcpy(chromosome.recessive, b.chromosome.recessive, Artist::genome_length);

  /* Set the fitness as high as possible to ensure it's replaced. */
  // fitness = std::numeric_limits<double>::max();
  fitness = 0;
  expected_reproduction = 1.0; /* This will be overwritten later */

  /* Copy A's location index. Don't increment. */
  location_index = a.location_index;

  /* Babies are not clones */
  is_clone = false;
}

/* Copy constuctor */
Artist::Artist(Artist const &that)
{
  fitness = that.fitness;
  location_index = that.location_index;
  expected_reproduction = that.expected_reproduction;

  is_clone = true;

  /* Allocate new memory for the new chromosome */
  chromosome.dominant = (uint8_t *) malloc(Artist::genome_length);
  chromosome.recessive = (uint8_t *) malloc(Artist::genome_length);

  /* Copy the bits into the new chromosome */
  memcpy(chromosome.dominant, that.chromosome.dominant, Artist::genome_length);
  memcpy(chromosome.recessive, that.chromosome.recessive, Artist::genome_length);
}

/* Assignment constructor */
void Artist::operator=(Artist const &that)
{
  fitness = that.fitness;
  location_index = that.location_index;
  expected_reproduction = that.expected_reproduction;

  /* Allocate new memory for the new chromosome */
  chromosome.dominant = (uint8_t *) malloc(Artist::genome_length);
  chromosome.recessive = (uint8_t *) malloc(Artist::genome_length);

  /* Copy the bits into the new chromosome */
  memcpy(chromosome.dominant, that.chromosome.dominant, Artist::genome_length);
  memcpy(chromosome.recessive, that.chromosome.recessive, Artist::genome_length);
}

/* Destructor */
Artist::~Artist()
{
  free(chromosome.dominant);
  free(chromosome.recessive);
}

/* * * * * * * * * * * * * * * * * * *
 *         Other Operators           *
 * * * * * * * * * * * * * * * * * * */

/* Allows us to sort without copying */
bool Artist::operator <(const Artist &a) const
{
  return fitness < a.fitness;
}

/* * * * * * * * * * * * * * * * * * *
 *         Member Functions          *
 * * * * * * * * * * * * * * * * * * */

/* Take a random double between [0,1] - if lower than or equal to 
  crossover_chance, swap part of the dominant and recessive genomes. The index
  to swap from is draw from an equal distribution from the 0th bit to the 
  last bit.
 */
void Artist::crossover()
{
  /* Only crossover $(crossover_chance)% of the time */
  double rand_zero_to_one = (double)(rand()/RAND_MAX);
  if(rand_zero_to_one < Artist::crossover_chance)
  {
      /* Where in the genome to crossover at */
      size_t bit_index = 0;
      size_t byte_index = 0;
      
      /* Choose a random byte to begin crossover in the genome. If XOVER_TYPE 
         is set to TRIANGLE - make sure that byte is aligned to the start of a
         Triangle struct.
       */
      if(Artist::crossover_type == Xover_type::TRIANGLE)
      {
        /* Generate a random, Triangle aligned, byte index */
        double div = (RAND_MAX/(Artist::number_of_triangles));
        size_t tri_index = (size_t)(std::floor((((double)rand() - 1)/div)));
        byte_index = TRIANGLE_LIST_BEGIN + (tri_index * sizeof(Triangle));
      }
      else
      {
        /* Generate a random byte index */
        double div = (RAND_MAX/(Artist::genome_length));
        byte_index = (size_t)(std::floor((((double)rand() - 1)/div)));
      }

      /* If applicable - choose a bit to crossover at */ 
      if(Artist::crossover_type == Xover_type::BIT)
      {
        /* Pick a bit to crossover at */
        double div = (RAND_MAX/(8 - 1));
        bit_index = (size_t)(std::floor((((double)rand() - 1)/div))); 
      }

      /* Swap the bytes in the range: [(byte_index + 1), genome_length].      
         There are no bytes to copy when crossover happens in the last byte.
       */
      if(byte_index <= (Artist::genome_length - 1))
      {
        /* Number of bytes to swap */
        size_t swap_size = Artist::genome_length - byte_index - 1;
        uint8_t * byte_swap_space = (uint8_t *)malloc(sizeof(uint8_t) * swap_size);
        memcpy(byte_swap_space, (chromosome.dominant + byte_index + 1), swap_size);
        memcpy((chromosome.dominant + byte_index + 1), (chromosome.recessive + byte_index + 1), swap_size);
        memcpy((chromosome.recessive + byte_index + 1), byte_swap_space, swap_size);

        /* Don't forget to clean up */
        free(byte_swap_space);
      }

      /* Swap the bits */
      if(Artist::crossover_type == Xover_type::BIT)
      {
        /* For the byte that is split, swap the bits */
        uint8_t dom_byte = chromosome.dominant[byte_index];
        uint8_t rec_byte = chromosome.recessive[byte_index];
        uint8_t intra_byte_index = bit_index % 8;
        uint8_t num_bits = 8 - intra_byte_index;

        /* Prepare the bytes for writing to */
        uint8_t mask1 = (uint8_t)GETMASK(intra_byte_index, num_bits);
        chromosome.dominant[byte_index] |= mask1;
        chromosome.recessive[byte_index] |= mask1;

        /* Prepare the bits to be written */
        uint8_t dom_bits = (dom_byte & mask1) | (~mask1);
        uint8_t rec_bits = (rec_byte & mask1) | (~mask1);

        /* Write the bits to the byte */
        chromosome.dominant[byte_index] &= rec_bits;
        chromosome.recessive[byte_index] &= dom_bits;
      }
  }
}

/* Chance to flip some of the bits */
void Artist::mutate()
{
  /* Create a binomial distribution to detect the number of mutations */
  size_t num_bits = 8 * genome_length * 2; /* 2 because diploid chromosome */
  std::binomial_distribution<size_t> distribution(num_bits, MUTATION_RATE);
  size_t num_mutations = distribution(Artist::rand_engine);

  /* Selecte a random bit to flip per mutation. */
  std::uniform_int_distribution<size_t> bit_selector(0, (num_bits - 1));
  for(size_t i = 0; i < num_mutations; i++)
  {
    /* Indices into the bits & bytes */
    size_t bit_index = bit_selector(Artist::rand_engine);
    size_t byte_index = bit_index / 8;
    uint8_t intra_byte_index = bit_index % 8;

    /* Select which genome to operate on. Dominant|Recessive */
    uint8_t * genome = (byte_index >= genome_length) ? chromosome.dominant : chromosome.recessive;
    byte_index %= genome_length;

    /* Flip the bit */
    uint8_t mask = (uint8_t)GETMASK(intra_byte_index, 1);
    genome[byte_index] ^= mask;
  }
}

/* Expresses the genotype, compares it to the submitted image and scores
 * it based on similarity. Sets and returns the fitness.
 */
void Artist::score()
{
  /* Express the phenotype */
  Magick::Image * canvas = draw();
  
  /* Compare the average pixel error to the original */
  double rmse = canvas->compare(Artist::source_copy, Magick::RootMeanSquaredErrorMetric);
  fitness = 1 / rmse;

  /* Free the memory */
  delete canvas;
}

/* Draw and return the image. Creates a NEW image that the caller is 
   responsible for freeing.
 */
Magick::Image * Artist::draw()
{
    /* Extract source iamge dimensions */
    size_t width = Artist::source_copy.columns();
    size_t height = Artist::source_copy.rows();

    /* Extract background color */
    uint8_t * rgba = chromosome.dominant + BG_COLOR_OFFSET;
    Magick::Color bg_color(rgba[0], rgba[1], rgba[2], rgba[3]);

    /* Create a canvas to draw upon. */
    std::string image_dimensions = std::to_string(width) + "x" + std::to_string(height);
    Magick::Image * canvas = new Magick::Image(image_dimensions, bg_color);

    /* Set some drawing defaults on the image */
    canvas->strokeWidth(0.0);
    canvas->strokeAntiAlias(true);

    /* Create a list of triangles to draw to the canvas. */
    Triangle * dominant = (Triangle *)(chromosome.dominant + TRIANGLE_LIST_BEGIN);
    Triangle * recessive = (Triangle *)(chromosome.recessive + TRIANGLE_LIST_BEGIN);
    std::vector<Magick::Drawable> triangle_list;

    for(size_t i = 0; i < GENOME_LENGTH; i++)
    {
      /* If we're in OAAT mode - limit expression */
      if(i >= Artist::expression_limit) { break; }

      Triangle const & tri_dom = dominant[i];
      Triangle const & tri_rec = recessive[i];
      
      Triangle tri = tri_dom.visible ? tri_dom : tri_rec;
      
      /* If both triangles were visibilty 0; draw neither */
      if(tri.visible) { continue; }
      
      /* Translate the coordinates */
      Magick::CoordinateList coordinates;
      coordinates.push_back(Magick::Coordinate(UITD(tri.x1, width), UITD(tri.y1, height)));
      coordinates.push_back(Magick::Coordinate(UITD(tri.x2, width), UITD(tri.y2, height)));
      coordinates.push_back(Magick::Coordinate(UITD(tri.x3, width), UITD(tri.y3, height)));
      Magick::DrawablePolyline drawable_triangle(coordinates);

      /* Set the fill/stroke color */
      Magick::Color color(tri.r, tri.g, tri.b, tri.a);
      canvas->strokeColor(color);
      canvas->fillColor(color);

      /* Push the colors and then the shape onto the drawable lsit */
      triangle_list.push_back(Magick::DrawableFillColor(color));
      triangle_list.push_back(Magick::DrawableStrokeColor(color));
      triangle_list.push_back(drawable_triangle);
    }

    /* Draw the triangles! */
    canvas->draw(triangle_list);

    return canvas;
}

/* * * * * * * * * * * * * * * * * * *
 *              Getters              *
 * * * * * * * * * * * * * * * * * * */

/* Returns the fitness of the Artist #getters */
double Artist::getFitness() const
{
  return fitness;
}

/* Returns the location index of the Artist. #getters */
size_t Artist::getLocationIndex() const
{
  return location_index;
}

/* Returns the expected_reproduction of the Artist. #getters */
double Artist::getExpectedReproduction() const
{
  return expected_reproduction;
}

/* Returns true if the artist is a clone. False otherwise. */
bool Artist::isClone() const
{
  return is_clone;
}

/* Sets is_clone to true */
void Artist::makeClone()
{
  is_clone = true;
}

/* Gets the max number of triangles artists are allowed to express. */
size_t Artist::getExpressionLimit()
{
  return expression_limit;
}

/* * * * * * * * * * * * * * * * * * *
 *              Setters              *
 * * * * * * * * * * * * * * * * * * */

/* Set the location index - in case it already exists */
void Artist::setLocationIndex(size_t index)
{
  if(index >= POPULATION_SIZE)
  {
    std::cerr << "Tried setting an out of bounds location index!\nIndex given: "
    << index << "\nAllowed indices: [0," << (POPULATION_SIZE - 1) << "]" << std::endl;
    exit(1);
  }
  else { location_index = index; }
}

/* Sets the proportion the artists should reproduce */
void Artist::setReproductionProportion(double avg_fitness, double std_dev)
{
  if(std_dev == 0) { expected_reproduction = 1; }
  else 
  {
    expected_reproduction = 1 + ((fitness - avg_fitness) / (2 * std_dev));
    /* Everyone has a smöl chance to make it */
    if(expected_reproduction <= 0) { expected_reproduction = 0.1; }
  }
}

/* Sets the max number of triangles artists are allowed to express. */
void Artist::setExpressionLimit(size_t n)
{
  if(n > number_of_triangles) { n = number_of_triangles; }
  expression_limit = n;
}

/* Increments the expression limit by 1 (if possible) */
void Artist::incrementExpressionLimit()
{
  if(expression_limit < number_of_triangles)
  {
    expression_limit++;
  }
}
