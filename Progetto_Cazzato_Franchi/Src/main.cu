#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <assert.h>
#include "warp_sort.cu"

int main(int argc, char *argv[]) {
	const int E = atoi(argv[1]), N = (1 << E);
	
	printf("N: %d (2^%d)\n", N, E);
	long long start = now();
		
	int *input1 = (int*) malloc(N * sizeof(int));
	int *input2 = (int*) malloc(N * sizeof(int));
	randInput(input1, N);
	memcpy(input2, input1, N * sizeof(int));
		
	printf("%-22s%lld ms\n","   Generazione input: ", now() - start);
		
	start = now();
		
	parallelSort(input1, N);
		
	printf("%-22s%lld ms\n","   Parallelo: ", now() - start);
		
	start = now();
		
	sort(input2, input2 + N);
		
	printf("%-22s%lld ms\n","   Sequenziale: ", now() - start);
	printf("\n");
		
	for(int i = 0; i < N; i ++)
		assert(input1[i] == input2[i]);
		
	free(input1);
	free(input2);
	
	return 0;
}
