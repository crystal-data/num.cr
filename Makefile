CC = clang
CFLAGS = -Ofast -march=native -Wall -std=c99

ext: kmeans random

kmeans: src/c/kmeans/asa136.c
	$(CC) -c src/c/kmeans/asa136.c -o src/c/kmeans/asa136.o -I. $(CFLAGS)

random: src/c/random/ranlib.c
	$(CC) -c src/c/random/ranlib.c -o src/c/random/ranlib.o -I. $(CFLAGS)
