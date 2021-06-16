#include <iostream>
#include <algorithm>
#include <bits/stdc++.h>
#include <random>
#include <chrono>

using namespace std;
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const int MAX_NUM = (1 << 25);
const int INF = MAX_NUM + 1;

void randInput(int *a, const int N) {
	for(int i = 0; i < N; i ++) 
		a[i] = (int)(rng() % MAX_NUM);
}

// Le funzioni "check" sono state utilizzate durante l'implementazione per verificare 
// la correttezza di ogni fase individualmente, mediante "assert"

bool checkBlocks(int *a, const int N, const int W) {
	for(int i = 0; i < N / W; i ++) 
		for(int j = i * W + 1; j < (i + 1) * W; j ++)
			if(a[j] < a[j - 1])
				return false;
	return true;
}

bool checkSplits(int *a, const int N, int *splitters, int *splitPoints, const int L, const int S) {
	const int SEQ_SIZE = N / L;
	for(int i = 0; i < L; i ++) {
		for(int z = 0; z <= splitPoints[i * (S + 1)]; z ++)
			if(a[i * SEQ_SIZE + z] > splitters[0])
				return false;
		for(int j = 1; j < S; j ++) 
			for(int z = splitPoints[i * (S + 1) + j - 1] + 1; z <= splitPoints[i * (S + 1) + j]; z ++) 
				if(a[i * SEQ_SIZE + z] <= splitters[j - 1] || a[i * SEQ_SIZE + z] > splitters[j])
					return false;
		for(int z = splitPoints[i * (S + 1) + S] + 1; z < SEQ_SIZE; z ++)
			if(a[i * SEQ_SIZE + z] <= splitters[S - 1])
				return false;
	}
	return true;
}

bool checkLastIn(int *a, const int N, int *lastIn, const int LAST_IN_SIZE) {
	if(lastIn[LAST_IN_SIZE - 1] != N - 1)
		return false;
	for(int i = 0; i < LAST_IN_SIZE; i ++) 
		for(int j = i > 0 ? lastIn[i - 1] + 1 : 0; j < lastIn[i]; j ++)
			if(a[j + 1] < a[j])
				return false;
	return true;
}

bool checkFinalMerge(int *a, const int N) {
	for(int i = 1; i < N; i ++)
		if(a[i] < a[i - 1])
			return false;
	return true;
}

long long now() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
