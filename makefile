NAME = triangles
OBJS = params.o lodepng.o image.o

# Determine which OS we're on to determine the compiler
OS := $(shell uname -s)
ifeq ($(OS), Darwin)
        CC = clang++
endif
ifeq ($(OS), Linux)
        CC = nvcc 
endif

DEBUG = -O3
CCFLAGS = -Wall -std=c++11 -c $(DEBUG)

triangles: main.o $(OBJS)
	$(CC) main.o $(OBJS) $(DEBUG) -o $(NAME)

main.o: main.h main.cpp params.h lodepng.h image.h
	$(CC) $(CCFLAGS) $(IMAGE_MAGICK) main.cpp

params.o: params.h params.cpp
	$(CC) $(CCFLAGS) params.cpp

lodepng.o: lodepng.cpp lodepng.h
	$(CC) $(CCFLAGS) lodepng.cpp

image.o: image.cpp image.h
	$(CC) $(CCFLAGS) image.cpp
  
# Utility
clean: 
	rm -f *.o $(NAME) $(TEST_NAME) && clear 2> /dev/null
