#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include "book.h"
#include "FinalProjectHelpers.h"


char playerInput[100];


__global__ void setup_kernel(curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void NonPoolingKernel(curandState *state, unsigned int *resultsArray, int totalBlocks) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id = y * totalBlocks + x;
	unsigned int values[3];
	
    // Copy state to local memory for efficiency
    curandState localState = state[id];
	
    // Generate pseudo-random unsigned ints
    values[0] = curand(&localState) % 10;
    values[1] = curand(&localState) % 10;
    values[2] = curand(&localState) % 10;
	
	for (int k = 0; k < 3; k++) {
		if (values[k] < 7) values[k] = 0;
		else if (values[k] < 9) values[k] = 1;
		else if (values[k] < 10) values[k] = 2;
	}
	
	int isMachineAWinner = 0;
	
	if (values[0] == values[1] && values[1] == values[2]) isMachineAWinner = 1;
	
	// Calculate prize value based upon symbols showing
	int prizeValue = 0;
	if (isMachineAWinner) {
		switch(values[0]) {
			case 0:
				prizeValue = 3;
				break;
			case 1:
				prizeValue = 10;
				break;
			case 2:
				prizeValue = 50;
				break;
		}
	}
	
	// TODO: Have each machine (thread) determine the level of winnings. Either pass in an array/"dictionary" or hard code it in the kernel.
	
    // Copy state back to global memory
    state[id] = localState;
	
    // Store results
    // TODO: Figure out what the symbol ratios should be. Percentages that add up to 100 would be simple
	resultsArray[id] = prizeValue;
}

__global__ void PoolingKernel(curandState *state, unsigned int *resultsArray, int totalBlocks) {
	// TODO: Similar to the NonPoolingKernel, but each block will have a thread for each machine in the pool and need to coordinate data
}

int main( void ) {
	PrintEmptyLines(50);
	
	// Game intro and explanation
	printf("Welcome to Turbo Gambling!\n\n");
	printf("Most of our services are under construction. The only game available is Turbo Slot Machines\n");
	
	DisplayRules();
	
	
	// START GAME (LOOP)
	int numSlotMachines = GetNumberOfMachinesInput(playerInput);
	
	int MACHINES_BY_FIVE_HUNDREDS = numSlotMachines / 500;
	
	// The player pressing Enter for the above input is read in the next input handling as ' ' for some reason...
	scanf("%c", playerInput);
	
	
	// Player Input for Pooling Machines or not
	//int isUserPoolingMachines = GetUserPoolingMachinesInput(playerInput);

	// Backend RNG Setup
	const unsigned int blocks = MACHINES_BY_FIVE_HUNDREDS;
	const unsigned int threads = 500;
	const unsigned int totalThreads = MACHINES_BY_FIVE_HUNDREDS * 500;
	
	curandState *devStates;
	unsigned int *devResults, *hostResults;
	
	// Allocate space for results on host
    hostResults = (unsigned int *)calloc(totalThreads, sizeof(int));

    // Allocate space for results on device
    HANDLE_ERROR(cudaMalloc((void **)&devResults, totalThreads * sizeof(unsigned int)));

    // Set results to 0
    HANDLE_ERROR(cudaMemset(devResults, 0, totalThreads * sizeof(unsigned int)));
	
	HANDLE_ERROR(cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState)));
	

	// Initialize Host and Device arrays
	//printf("Size: %i \n", totalThreads);
	unsigned int numbytes = totalThreads * sizeof(unsigned int);
	//printf("Numbytes: %i \n", numbytes);
	unsigned int *resultsArray = (unsigned int *) malloc(numbytes);
	unsigned int *dev_resultsArray = 0;
	
	HANDLE_ERROR(cudaMalloc((void**)&dev_resultsArray, numbytes));
	
	setup_kernel<<<blocks, threads>>>(devStates);
	
	// If Pool then Player Input for how to partition them
	//if (isUserPoolingMachines) {
		//PoolingKernel<<<blocks, threads>>>();
	//}
	//else {
		NonPoolingKernel<<<blocks, threads>>>(devStates, dev_resultsArray, MACHINES_BY_FIVE_HUNDREDS);
	//}
	
	HANDLE_ERROR(cudaMemcpy(resultsArray, dev_resultsArray, totalThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	int gamesWon = 0;
	int zeroWinners = 0;
	int oneWinners = 0;
	int twoWinners = 0;
	for (int k = 0; k < totalThreads; k++) {
		//printf("%i ", resultsArray[k]);
		if (resultsArray[k] > 0) gamesWon++;
		if (resultsArray[k] == 3) zeroWinners++;
		if (resultsArray[k] == 10) oneWinners++;
		if (resultsArray[k] == 50) twoWinners++;
	}
	printf("\n");
	printf("Number of machines that were winners: %i\n", gamesWon);
	printf("Number of zero winners: %i\n", zeroWinners);
	printf("Number of one winners: %i\n", oneWinners);
	printf("Number of two winners: %i\n", twoWinners);
	
	// OUTPUT RESULTS
	//printf("Number of winnings machines: %i\nNumber of losing machines: %i\nMoney gained/lost: %i", wonMachines, lostMachines, -1 * moneySpent + moneyWon);
	
	// CLEAN-UP
	free(resultsArray);
	HANDLE_ERROR(cudaFree(dev_resultsArray));
	
	HANDLE_ERROR(cudaFree(devStates));
	HANDLE_ERROR(cudaFree(devResults));
    free(hostResults);

	/*
	HIGH LEVEL TODO LIST
	------------------------
	- Cuda Work
		- None pooling kernel
		- Pooling kernel
	- Processing machine results
		- (Minimum) Display # of machines that won and total winnings, maybe some other user data
		- (Optional) Display some pictoral symbols, possibly of the best winning machine
		- (Optional) Display the rolling of symbols, "
	- Timing and stats data
	*/

	PrintEmptyLines(2);

	return 0;
}