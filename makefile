NAME = triangles
OBJS = params.o lodepng.o image.o

# Determine which OS we're on to determine the compiler
OS := $(shell uname -s)
ifeq ($(OS), Darwin)
        HOST_COMPILER = clang++
endif
ifeq ($(OS), Linux)
        HOST_COMPILER = nvcc 
endif

DEBUG = -O3
CCFLAGS = -std=c++11 -c $(DEBUG)

NVCC:= $(HOST_COMPILER)

triangles: main.o $(OBJS)
	$(NVCC) main.o $(OBJS) $(DEBUG) -o $(NAME)

main.o: main.h main.cpp params.h lodepng.h image.h
	$(NVCC) $(CCFLAGS) $(IMAGE_MAGICK) main.cpp

params.o: params.h params.cpp
	$(NVCC) $(CCFLAGS) params.cpp

lodepng.o: lodepng.cpp lodepng.h
	$(NVCC) $(CCFLAGS) lodepng.cpp

image.o: image.cpp image.h
	$(NVCC) $(CCFLAGS) image.cpp
  
# Utility
clean: 
	rm -f *.o $(NAME) $(TEST_NAME) && clear 2> /dev/null
