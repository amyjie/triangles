NAME = triangles
OBJS = params.o 

# Determine which OS we're on to determine the compiler
OS := $(shell uname -s)
ifeq ($(OS), Darwin)
        CC = clang++
endif
ifeq ($(OS), Linux)
        CC = g++
endif

DEBUG = -O3
CCFLAGS = -Wall -std=c++11 -c $(DEBUG)

triangles: main.o $(OBJS)
	$(CC) main.o $(OBJS) $(DEBUG) -o $(NAME)

main.o: main.h main.cpp params.h 
	$(CC) $(CCFLAGS) $(IMAGE_MAGICK) main.cpp

params.o: params.h params.cpp
	$(CC) $(CCFLAGS) params.cpp

# Utility
clean: 
	rm -f *.o $(NAME) $(TEST_NAME) && clear 2> /dev/null
