#include "main.h"

int main(int argc, char ** argv)
{	
	/* Turns command line arguments into hyper parameters */
	parseArgs(argc, argv);
	/* Allows use of ImageMagick */
	Magick::InitializeMagick(*argv);

	/* Initialize Artist settings with runtime parameters. */
	Artist::initialization(GENOME_LENGTH, MUTATION_RATE, XOVER_CHANCE, RANDOM_SEED);

	/* Precompute locations and distances between them */
	auto location_likelihood_map = getLocationLikelihoodMap(POPULATION_SIZE);

	/* Open the source Image */
	Magick::Image source = openImage(IMAGE_PATH);

	/* Initialize the srand for crossover/mutation decisions */
	srand(RANDOM_SEED);

	/* Generate a list of Artists */
	std::vector<Artist *> artists;
	artists.reserve(POPULATION_SIZE); /* Save us reallocating a few times */
	for(size_t i = 0; i < POPULATION_SIZE; i++)
	{
		artists.push_back(new Artist());
	}

	/* Main loops - runs for # of GENERATIONS */
	size_t number_of_generations_run = 0;  /* Keep track of which generation we're on */
	bool run_forever = (GENERATIONS == 0); /* If # of generations is 0 -> run forever */
	for(;number_of_generations_run < GENERATIONS || run_forever; number_of_generations_run++)
	{
		/* Add the artists to a location indexed vector so that it's easy to 
		   look them up later for mating. If a space is already occupied, it
		   will search for the nearest empty location to fill.
		*/
		std::vector<Artist *> location_map(POPULATION_SIZE);
		std::fill(location_map.begin(), location_map.end(), (Artist *)NULL);
		for(auto a = artists.begin(); a != artists.end(); ++a)
		{
			bool unsuccessful = true;
			size_t location_index = (*a)->getLocationIndex();
			while(unsuccessful)
			{
				/* If the location is available - add the Artist there */
				if(location_map[location_index] == NULL)
				{
					unsuccessful = false;
					location_map[location_index] = (*a);
				}
				else /* Find a new location based on the precomputed map */
				{

				}
			}
			// location_map[i] = (*a);
		}

		/* Run through the list of artists, perform crossover, mutate them and
		   and score them. [ thread this later ]
		 */
		for(auto a = artists.begin(); a != artists.end(); ++a)
		{
			(*a)->crossover();
			(*a)->mutate();
			(*a)->score(source);
		}

		/* Sort the artists from best to worst */
		std::sort(artists.begin(), artists.end(), [](const Artist * a, const Artist * b) -> bool { 
        	return a->getFitness() < b->getFitness(); 
    	});

		/* Calculate the average fitness & std dev of the artists */
		double std_dev = 0.0;
		double avg_fitness = 0.0;
		for(auto a = artists.begin(); a != artists.end(); ++a)
		{
			avg_fitness += (*a)->getFitness();
		}
		avg_fitness /= artists.size();

		/* Calculate the std dev of the artists */
		for(auto a = artists.begin(); a != artists.end(); ++a)
		{
			std_dev += pow((*a)->getFitness() - avg_fitness, 2);
			std_dev = sqrt(std_dev/POPULATION_SIZE);
		}

		/* Calculate the number of times an artist should reproduce */
		for(auto a = artists.begin(); a != artists.end(); ++a)
		{
			(*a)->setReproductionProportion(avg_fitness, std_dev);
		}

		/* Add artists to a new vector, in proportion to their fitness, with
		   a small amount of randomness for good measure.
		 */
		std::vector<Artist *> artists_proportional;
		artists_proportional.reserve(POPULATION_SIZE);

		double roulette = (double) ((double)rand()/(double)RAND_MAX);
		double sum = 0.0;
		size_t index = 0;
		for(;artists_proportional.size() < artists.size();)
		{
			double expected_reproduction = artists[index]->getExpectedReproduction();
			for(sum += expected_reproduction; sum > roulette; roulette++)
			{
				artists_proportional.push_back(artists[index]);
			}
			index++;
		}
		artists_proportional.resize(POPULATION_SIZE);

		/* Mate the artists to produce the next generation in proportion to
		   their fitness.
		 */
		for(auto a = artists_proportional.begin(); a != artists_proportional.end(); ++a)
		{
		}



	}


}
