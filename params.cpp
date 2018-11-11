#include "params.h"

/* The image we're operating on */
char * IMAGE_PATH;

/* Maximum number of triangles to attempt to draw the image with. */
size_t GENOME_LENGTH = 10;

/* Number of artists in each generation */
size_t POPULATION_SIZE = 25;

/* Number of children the population produces each generation */
size_t NUM_CHILDREN = 10;

/* Amount of effort to put into finding a solution */
size_t EFFORT = 10000; 

/* srand() seed for repeatable testing */
unsigned int RANDOM_SEED = time(NULL);

/* Chance for a diploid chromosome to crossover. */
double XOVER_CHANCE = 0.7;

/* Where the genome is allowed to crossover */
Xover_type XOVER_TYPE = Xover_type::TRIANGLE;

/* Chance, per bit, of being flipped each generation. Defaults to 0.005. */
double MUTATION_RATE = 0.005;

/* Chance to swap two triangles, or randomize a triangle. */
double MACRO_MUTATION_RATE = 0.5;

void parseArgs(int argc, char ** argv)
{
    int c;
    while(true)
    {
        static struct option long_options[] =
        {
            {"image",               required_argument, 0, 'i'},
            {"effort",              required_argument, 0, 'e'},
            {"genome-length",       required_argument, 0, 'g'},
            {"population-size",     required_argument, 0, 'p'},
            {"number-of-children",  required_argument, 0, 'c'},
            {"random-seed",         required_argument, 0, 'r'},
            {"mutation-rate",       required_argument, 0, 'm'},
            {"macro-mutation-rate", required_argument, 0, 'b'},
            {"crossover-type",      required_argument, 0, 't'},
            {"crossover-chance",    required_argument, 0, 'x'},
            {0, 0, 0, 0}
        };

        /* getopt_long stores the option index here. */
        int option_index = 0;

        /* Short codes for characters */
        const char * short_options = "i:e:g:p:c:r:m:b:t:x:";

        c = getopt_long(argc, argv, short_options, long_options, &option_index);

        /* End of CLI arguments */
        if(c == -1) { break; }

        /* Almighty switch statement */
        switch(c)
        {
            case 0:
                if(long_options[option_index].flag != 0)
                    break;
                printf("option %s", long_options[option_index].name);
                if(optarg) { printf(" with arg %s", optarg); }
                printf("\n");
                break;

            case 'i': {
                IMAGE_PATH = strdup(optarg);
                break;
            }
            case 'g': {
                int g = atoi(optarg);
                if(g < 0)
                {
                    printf("The genome length must be greater than 0.\n");
                    exit(1);
                }
                else
                {
                    GENOME_LENGTH = g;
                }                
                break;
            }
            case 'c': {
                int c = atoi(optarg);
                if(c < 0)
                {
                    printf("The number of children must be greater than 0.\n");
                    exit(1);
                }
                else
                {
                    NUM_CHILDREN = c;
                }                
                break;
            }
            case 'p': {
                int p = atoi(optarg);
                if(p < 0)
                {
                    printf("The population size must be greater than 0.\n");
                    exit(1);
                }
                else
                {
                    POPULATION_SIZE = p;
                }                
                break;
            }
            case 'e': {
                int e = atoi(optarg);
                if(e < 0)
                {
                    printf("Effort must be greater than 0.\n");
                    exit(1);
                }
                else
                {
                    EFFORT = e;
                }
                break;
            }
            case 'r': {
                RANDOM_SEED = atoi(optarg); 
                break;
            }
            case 'x': {
                XOVER_CHANCE = std::stod(optarg);
                if(XOVER_CHANCE < 0 || XOVER_CHANCE > 1)
                { 
                    printf("Crossover chance must be between 0.0 and 1.0.\nChance provided:%f", XOVER_CHANCE);
                    exit(1);
                }
                break;
            }
            case 't': {
                std::string type(optarg);
                std::transform(type.begin(), type.end(), type.begin(), ::tolower);

                if(type == "bit") { XOVER_TYPE = Xover_type::BIT; }
                else if(type == "byte") { XOVER_TYPE = Xover_type::BYTE; }
                else if(type == "triangle") { XOVER_TYPE = Xover_type::TRIANGLE; }
                else
                { 
                    printf("Crossover type must be one of bit, byte, triangle.\nType provided: %s\n", optarg);
                    exit(1);
                }
                break;
            }
            case 'm': {
                MUTATION_RATE = std::stod(optarg);
                if(MUTATION_RATE < 0 || MUTATION_RATE > 1)
                { 
                    printf("Mutation rate must be between 0.0 and 1.0.\nRate provided:%f", MUTATION_RATE);
                    exit(1);
                }
                break;
            }
            case 'b': {
                MACRO_MUTATION_RATE = std::stod(optarg);
                if(MACRO_MUTATION_RATE < 0 || MACRO_MUTATION_RATE > 1)
                { 
                    printf("Macro mutation rate must be between 0.0 and 1.0.\nRate provided:%f", MACRO_MUTATION_RATE);
                    exit(1);
                }
                break;
            }
            case '?': {
                break;
            }
            default: {
                abort();
            }
        }
    }

    /* Image path is the default argument */
    if(optind < argc)
    {
        IMAGE_PATH = argv[optind];
    }

    /* Image path is the only required argument */
    if(!IMAGE_PATH)
    {
        printf("You must supply a path to an image using "
               "triangles -i \"path/to/image\"\n");
        exit(1);
    }
}
