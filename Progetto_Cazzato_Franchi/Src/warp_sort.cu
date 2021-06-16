#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.cpp"
#include <assert.h>

// Assunzioni: W = 2^m (m >= 7), T = 2^r (r >= 6), S = 2^z (z >= 6).
// Ad ogni modo, WARP_PER_BLK * W e WARP_PER_BLK * T non possono eccedere |SMEM|,
// quindi sono abbastanza limitati.
// Inoltre: T / 2 multiplo di WARP_SIZE
const int W = (1 << 7);
const int T = (1 << 6); 
const int L = (1 << 6);
const int K = (1 << 6); // Non c'è un valore migliore a prescindere. Si tratta di un'euristica
const int S = (1 << 8); 
const int WARP_SIZE = (1 << 5);
const int MAX_TH_PER_BLK = (1 << 10);
const int MAX_WARP_PER_BLK = MAX_TH_PER_BLK / WARP_SIZE;

__device__ __managed__ int *a;
__device__ __managed__ int *b;
__device__ __managed__ int *splitters;
__device__ __managed__ int *splitPoints; // splitPoints[i][j] = indice ultimo elemento sequenza i <= splitters[j]
__device__ __managed__ int *lastIna;
__device__ __managed__ int *lastInb;
__device__ __managed__ int *missingUpToJ;

// STEP 1

// Comparatori
__device__ void asc(int *a, int pos, int j) {
	// Più efficiente di un if (per evitare divergenza)
	int supp = a[pos];
	a[pos] = min(a[pos], a[pos + j]);
	a[pos + j] = max(supp, a[pos + j]);
}

__device__ void desc(int *a, int pos, int j) {
	int supp = a[pos];
	a[pos] = max(a[pos], a[pos + j]);
	a[pos + j] = min(supp, a[pos + j]);
}

// Viene usato sia da bitonicSort, sia da merge
__device__ void sortBuff(int *buff, const int SIZE, const int TH_NUM) {
	// Caso speciale: i = SIZE
	for(int z = 0; z < (SIZE >> 1) / WARP_SIZE; z ++) 
		asc(buff, TH_NUM + z * WARP_SIZE, SIZE >> 1);
	// Unica rampa: i = SIZE / 2
	for(int j = SIZE >> 2; j > 0; j >>= 1) {
		for(int z = 0; z < max(1, (SIZE >> 2) / WARP_SIZE); z ++) {
			const int K = TH_NUM + z * WARP_SIZE;
			const int RAMP_OFF = (K / j) * (j << 1) + (K % j);
			// Solo ascendenti
			asc(buff, RAMP_OFF, j); 
			if((SIZE >> 2) >= WARP_SIZE)
				asc(buff, RAMP_OFF + (SIZE >> 1), j);
		}
	}
}

__global__ void bitonicSort(int *a, const int N) {
	__shared__ int buffer[MAX_WARP_PER_BLK][W]; // max 49152 byte di smem
	// Ad ogni warp corrisponde un blocco di W elementi da ordinare
	const int WARP_NUM = threadIdx.x / WARP_SIZE;
	const int TH_NUM = threadIdx.x % WARP_SIZE;
	const int FROM = (blockIdx.x * (blockDim.x / WARP_SIZE) + WARP_NUM) * W; // Primo elemento del blocco da ordinare
	
	if(FROM >= N)
		return;
	
	int *buff = buffer[WARP_NUM];
	for(int z = 0; z < W / WARP_SIZE; z ++)
		buff[TH_NUM + z * WARP_SIZE] = a[FROM + TH_NUM + z * WARP_SIZE]; // Accesso coalescente
	
	// i indica la fase (ed è |rampa|), mentre j lo stage
	for(int i = 2; i < W; i <<= 1) {
		for(int j = i >> 1; j > 0; j >>= 1) {
			// Ogni rampa ha bisogno di i / 2 comparatori
			for(int z = 0; z < max(1, (W >> 2) / WARP_SIZE); z ++) {
				const int CURR = TH_NUM + z * WARP_SIZE;
				const int RAMP_FROM = CURR / (i >> 1) * (i << 1); 
				const int RAMP_OFF = (CURR % (i >> 1) / j) * (j << 1) + (CURR % (i >> 1) % j);
				asc(buff, RAMP_FROM + RAMP_OFF, j);
				desc(buff, RAMP_FROM + RAMP_OFF + i, j);
			}
		}
	}
	
	sortBuff(buff, W, TH_NUM);
	
	for(int z = 0; z < W / WARP_SIZE; z ++)
		a[FROM + TH_NUM + z * WARP_SIZE] = buff[TH_NUM + z * WARP_SIZE]; // Accesso coalescente
}


// STEP 2

__device__ void pick(int *src, int* dst, const int TO_PICK, const int TH_NUM, bool rev, int &offToMod) {
	if(rev)
		for(int z = 0; z < TO_PICK / WARP_SIZE; z ++)
			dst[TO_PICK - 1 - (TH_NUM + z * WARP_SIZE)] = src[TH_NUM + z * WARP_SIZE];
	else
		for(int z = 0; z < TO_PICK / WARP_SIZE; z ++)
			dst[TH_NUM + z * WARP_SIZE] = src[TH_NUM + z * WARP_SIZE];
	offToMod += TO_PICK;
}

__global__ void merge(int *a, int *b, const int SEQ_SIZE, const int N) {
	__shared__ int buffer[MAX_WARP_PER_BLK][T]; // max 49152 byte di smem
	const int WARP_NUM = threadIdx.x / WARP_SIZE;
	const int TH_NUM = threadIdx.x % WARP_SIZE;
	const int FROM = (blockIdx.x * blockDim.x / WARP_SIZE + WARP_NUM) * (SEQ_SIZE << 1);
	
	if(FROM >= N)
		return;
	
	const int TO_PICK = T >> 1;
	int *buff = buffer[WARP_NUM];
	int *A = a + FROM, *B = a + FROM + SEQ_SIZE;
	int offA = 0, offB = 0, off = 0;
	pick(B, buff + TO_PICK, TO_PICK, TH_NUM, false, offB);
	bool fromA = true;
	while(off < (SEQ_SIZE << 1) - TO_PICK) { // L'ultimo pezzo è la seconda metà di buff
		// Non c'è warp div. Tutti i thread del warp vanno o nell'if, o nell'else
		fromA ? pick(A + offA, buff, TO_PICK, TH_NUM, true, offA): 
				pick(B + offB, buff, TO_PICK, TH_NUM, true, offB);
				
		fromA = (offB == SEQ_SIZE || (offA < SEQ_SIZE && A[offA - 1] < B[offB - 1]));
		sortBuff(buff, T, TH_NUM);
		pick(buff, b + FROM + off, TO_PICK, TH_NUM, false, off);
	}
	pick(buff + TO_PICK, b + FROM + off, TO_PICK, TH_NUM, false, off);
	
	// Copiare in a
	for(int i = 0; i < (SEQ_SIZE << 1) / WARP_SIZE; i ++)
		a[FROM + i * WARP_SIZE + TH_NUM] = b[FROM + i * WARP_SIZE + TH_NUM];
}


// STEP 3

__device__ void binSearch(int *seq, const int SEQ_SIZE, int *splitters, int *splitPoints, const int SPLIT_ID, const int SEQ_ID) {
	int l = 0, r = SEQ_SIZE - 1, last = -1;
	while(l <= r) {
		int m = (l + r) >> 1;
		if(seq[m] <= splitters[SPLIT_ID]) {
			last = m;
			l = m + 1;
		} else 
			r = m - 1;
	}
	splitPoints[SEQ_ID * (S + 1) + SPLIT_ID] = last; // -1 sta per seq vuota
}

__global__ void split(int *a, int SEQ_SIZE, int *splitters, int *splitPoints) {
	const int WARP_NUM = threadIdx.x / WARP_SIZE;
	const int TH_NUM = threadIdx.x % WARP_SIZE;
	const int SEQ_ID = blockIdx.x * blockDim.x / WARP_SIZE + WARP_NUM;
	for(int z = 0; z < S / WARP_SIZE; z ++)
		binSearch(a + SEQ_ID * SEQ_SIZE, SEQ_SIZE, splitters, splitPoints, TH_NUM + z * WARP_SIZE, SEQ_ID); // Accesso coalescente
	// Caso speciale (seq S + 1)
	if(TH_NUM == 0) // S + 1 non è multiplo di WARP_SIZE
		splitPoints[SEQ_ID * (S + 1) + S] = SEQ_SIZE - 1;
}

__global__ void copyKeepingPadd(int *a, int *b, int *splitPoints, int *lastIn, const int SEQ_SIZE, const int WARPS_PER_SUB_SEQ) {
	const int SUB_SEQ_ID = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE / WARPS_PER_SUB_SEQ;
	
	if(SUB_SEQ_ID >= L * (S + 1))
		return;
	
	const int TH_NUM = threadIdx.x % WARP_SIZE;
	const int WARP_ID = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE % WARPS_PER_SUB_SEQ;
	const int I = SUB_SEQ_ID % L, J = SUB_SEQ_ID / L;
	const int FROM_A = I * SEQ_SIZE + (J > 0 ? splitPoints[I * (S + 1) + (J - 1)] + 1 : 0);
	const int FROM_B = SUB_SEQ_ID > 0 ? lastIn[SUB_SEQ_ID - 1] + 1 : 0;
	const int OFF = WARP_ID * WARP_SIZE;
	const int SUB_SEQ_LEN = J > 0 ? splitPoints[I * (S + 1) + J] - splitPoints[I * (S + 1) + (J - 1)] : 
									splitPoints[I * (S + 1)] + 1;
	for(int z = 0; OFF + TH_NUM + z * WARP_SIZE < SUB_SEQ_LEN; z += WARPS_PER_SUB_SEQ)
		b[FROM_B + OFF + TH_NUM + z * WARP_SIZE] = a[FROM_A + OFF + TH_NUM + z * WARP_SIZE];
}


// STEP 4

__global__ void step4(int *a, int *b, int *lastIna, int *lastInb, const int PAIRS_TO_MERGE) {
	const int WARP_NUM = threadIdx.x / WARP_SIZE;
	const int TH_NUM = threadIdx.x % WARP_SIZE;
	const int PAIR = blockDim.x * blockIdx.x / WARP_SIZE + WARP_NUM;
	
	if(PAIR >= PAIRS_TO_MERGE)
		return;
	
	// Non c'è warp div. Tutti i thread del warp vanno o nell'if, o nell'else
	const int FROM_A = PAIR > 0 ? lastIna[(PAIR << 1) - 1] + 1 : 0;
	const int FROM_B = lastIna[PAIR << 1] + 1;
	const int LEN_A = PAIR > 0 ? lastIna[PAIR << 1] - lastIna[(PAIR << 1) - 1] : lastIna[0] + 1;
	const int LEN_B = lastIna[(PAIR << 1) + 1] - lastIna[PAIR << 1];
	
	__shared__ int buffer[MAX_WARP_PER_BLK][T]; // max 49152 byte di smem
	const int TO_PICK = T >> 1;
	int *buff = buffer[WARP_NUM];
	int *A = a + FROM_A, *B = a + FROM_B;
	int offA = 0, offB = 0, off = 0;
	pick(B, buff + TO_PICK, TO_PICK, TH_NUM, false, offB);
	bool fromA = true;
	while(off < (LEN_A + LEN_B) - TO_PICK) { // L'ultimo pezzo è la seconda metà di buff
		// Non c'è warp div. Tutti i thread del warp vanno o nell'if, o nell'else
		fromA ? pick(A + offA, buff, TO_PICK, TH_NUM, true, offA): 
				pick(B + offB, buff, TO_PICK, TH_NUM, true, offB);
		
		fromA = (offB == LEN_B || (offA < LEN_A && A[offA - 1] < B[offB - 1]));
		sortBuff(buff, T, TH_NUM);
		pick(buff, b + FROM_A + off, TO_PICK, TH_NUM, false, off);
	}
	pick(buff + TO_PICK, b + FROM_A + off, TO_PICK, TH_NUM, false, off);
	
	// Non serve copiare in a perchè qui si usa inv
	// Le coppie dimezzano
	lastInb[PAIR] = lastIna[(PAIR << 1) + 1];
}

// La divergenza può avvenire solo a fine sequenza
__global__ void filter(int *A, int *B, const int ELEMS) {
	const int GLOB_WARP_NUM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
	const int TH_NUM = threadIdx.x % WARP_SIZE;
	const int WARPS = gridDim.x * blockDim.x / WARP_SIZE;
	const int OFF = GLOB_WARP_NUM * WARP_SIZE;
	for(int i = 0; OFF + TH_NUM + i * WARP_SIZE < ELEMS; i += WARPS) {
		if(A[OFF + TH_NUM + i * WARP_SIZE] == INF) 
			return;
		B[OFF + TH_NUM + i * WARP_SIZE] = A[OFF + TH_NUM + i * WARP_SIZE];
	}
}

// Un warp per ogni blocco da W elementi
__global__ void filterAll(int *a, int *b, int *lastIn, int *missingUpToJ) {
	const int SEQ_ID = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(SEQ_ID >= S + 1)
		return;
	
	const int FROM_A = (SEQ_ID > 0 ? lastIn[SEQ_ID - 1] + 1 : 0);
	const int FROM_B = (SEQ_ID > 0 ? lastIn[SEQ_ID - 1] + 1 - missingUpToJ[SEQ_ID - 1] : 0);
	const int ELEMS = (SEQ_ID > 0 ? lastIn[SEQ_ID] - lastIn[SEQ_ID - 1] : lastIn[0] + 1);
	// Il "+1" serve perchè in genere ELEMS non è multiplo di W * MAX_WARP_PER_BLK
	filter <<<ELEMS / (W * MAX_WARP_PER_BLK) + 1, MAX_TH_PER_BLK>>> (a + FROM_A, b + FROM_B, ELEMS);
}


void parallelSort(int *input, const int N) {
	// Assumiamo N potenza di 2 e maggiore o uguale a W e L
	cudaMallocManaged(&a, N * sizeof(int), cudaMemAttachGlobal);
	cudaMallocManaged(&b, N * sizeof(int), cudaMemAttachGlobal);
	cudaMallocManaged(&splitters, S * sizeof(int), cudaMemAttachGlobal);
	cudaMallocManaged(&splitPoints, L * (S + 1) * sizeof(int), cudaMemAttachGlobal);
	
	memcpy(a, input, N * sizeof(int));
	
	// PREAMBLE STEP 3
	int samples[S * K];
	for(int i = 0; i < S * K; i ++) 
		samples[i] = a[rng() % N];
	sort(samples, samples + S * K);
	for(int i = K - 1; i < S * K; i += K)
		splitters[i / K] = samples[i];
		
	// STEP 1
	bitonicSort <<<max(1, N / (MAX_WARP_PER_BLK * W)), MAX_TH_PER_BLK>>> (a, N);
	cudaDeviceSynchronize();
		
	// STEP 2
	for(int seq_size = W; seq_size < N / L; seq_size <<= 1) {
		merge <<<max(1, N / ((seq_size << 1) * MAX_WARP_PER_BLK)), MAX_TH_PER_BLK>>> (a, b, seq_size, N);
		cudaDeviceSynchronize();
	}
	
	// STEP 3
	// Per massimizzare il parallelismo (la GPU utilizzata ha 24 SM).
	// Assumendo L abbastanza piccolo (e in genere lo è: il paper consiglia >= 64)
	split <<<L, WARP_SIZE>>> (a, N / L, splitters, splitPoints);
	cudaDeviceSynchronize();
	
	// lastIn[i] indica l'ultimo indice della sottosequenza i-esima nel nuovo array 
	// b, ottenuto aggiungendo il padding. lastIna e lastInb perchè non si può lavorare 
	// in-place
	cudaMallocManaged(&lastIna, (S + 1) * L * sizeof(int), cudaMemAttachGlobal);
	cudaMallocManaged(&lastInb, (S + 1) * L * sizeof(int), cudaMemAttachGlobal);
	cudaMallocManaged(&missingUpToJ, (S + 1) * sizeof(int), cudaMemAttachGlobal);
	
	// Calcolo quantità di padding (sequenziale, tanto S e L sono costanti)
	int newsize = 0;
	for(int j = 0; j < S + 1; j ++) {
		if(j > 0)
			missingUpToJ[j] = missingUpToJ[j - 1];
		for(int i = 0; i < L; i ++) {
			int seqLen = j > 0 ? splitPoints[i * (S + 1) + j] - splitPoints[i * (S + 1) + (j - 1)] : 
								 splitPoints[i * (S + 1)] + 1;
			int missing = (T >> 1) - (seqLen % (T >> 1));
			newsize += seqLen + missing;
			lastInb[j * L + i] = newsize - 1;
			missingUpToJ[j] += missing;
		}
	}
		
	cudaMallocManaged(&b, newsize * sizeof(int), cudaMemAttachGlobal);
	for(int i = 0; i < newsize; i ++)
		b[i] = INF;
		
	// Idealmente un warp per ogni blocco di W elementi da spostare
	const int WARPS_PER_SUB_SEQ = N / (L * (S + 1)) / W + 1;
	copyKeepingPadd <<<L * (S + 1) * WARPS_PER_SUB_SEQ / MAX_WARP_PER_BLK, MAX_TH_PER_BLK>>> (a, b, splitPoints, lastInb, N / L, WARPS_PER_SUB_SEQ);
	cudaDeviceSynchronize();
	cudaMallocManaged(&a, newsize * sizeof(int), cudaMemAttachGlobal);
		
	// STEP 4
	bool inv = true;
	for(int pairs = (S + 1) * (L >> 1); pairs >= S + 1; pairs >>= 1) { // L è potenza di 2 e >= 64
		int blocks = pairs / MAX_WARP_PER_BLK + (pairs % MAX_WARP_PER_BLK != 0 ? 1 : 0); // S + 1 però è dispari
		if(inv)
			step4 <<<blocks, MAX_TH_PER_BLK>>> (b, a, lastInb, lastIna, pairs);
		else
			step4 <<<blocks, MAX_TH_PER_BLK>>> (a, b, lastIna, lastInb, pairs);
		cudaDeviceSynchronize();
		inv = !inv;
	}
		
	// L'array deve essere filtrato. Le S + 1 sequenze non vengono fuse ed il 
	// padding infrange la proprietà indotta dagli splitter.
	// Con S potenza di 2, S + 1 non è mai multiplo di MAX_WARP_PER_BLK
	if(inv)
		filterAll <<<(S + 1) / MAX_WARP_PER_BLK + 1, MAX_TH_PER_BLK>>> (b, a, lastInb, missingUpToJ);
	else
		filterAll <<<(S + 1) / MAX_WARP_PER_BLK + 1, MAX_TH_PER_BLK>>> (a, b, lastIna, missingUpToJ);
	inv = !inv;
	cudaDeviceSynchronize();
	
	// Il risultato si troverà nelle prime N celle di a
	if(inv)
		a = b;
		
	memcpy(input, a, N * sizeof(int));
	
	cudaFree(a);
	cudaFree(b);
	cudaFree(splitters);
	cudaFree(splitPoints);
	cudaFree(lastIna);
	cudaFree(lastInb);
	cudaFree(missingUpToJ);
}
