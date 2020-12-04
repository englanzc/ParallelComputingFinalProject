#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <time.h>
#include <string.h>
#include <chrono> 
#include "book.h"
#include "gputimer.h"
#include "FinalProjectHelpers.h"


char playerInput[100];


__global__ void setupRNG_kernel(curandState *state, long time) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
	// Each thread gets same seed, a different sequence number, no offset	
	curand_init(time, id, 0, &state[id]);
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
	
	// Assign the appropriate symbol based on percentage it should appear
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
	
    // Copy state back to global memory
    state[id] = localState;
	
    // Store results
	resultsArray[id] = prizeValue;
}

__global__ void PoolingIndividualMachinesKernel(curandState *state, unsigned int *resultsArray, int totalBlocks) {
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
	
	// Assign the appropriate symbol based on percentage it should appear
	for (int k = 0; k < 3; k++) {
		if (values[k] < 7) values[k] = 0;
		else if (values[k] < 9) values[k] = 1;
		else if (values[k] < 10) values[k] = 2;
	}
	
	// If this individual machine is a winner, then output what kind of winner it is {0, 1, 2} else -1 for loser
	int typeOfWinner = -1;
	if (values[0] == values[1] && values[1] == values[2]) { typeOfWinner = values[0]; }
	
	// Copy state back to global memory
    state[id] = localState;
	
    // Store results
	resultsArray[id] = typeOfWinner;
}

__global__ void PoolingProcessingKernel(unsigned int *resultsArray, unsigned int *pooledPrizeResults, int partition, int totalMachines) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id = y * 500 + x * partition;
	
	// There will be excess threads, so we do not want index the array outside its size
	if (id >= totalMachines) return;
	
	int currentSameWinnerType = -2;
	int areAllMachinesSameWinner = 1;
	int areAllMachinesAWinner = 1;
	int previousMachinesResult = -2;
	int allDifferentWinnersPrizeTotal = 0;
	int notAllWinnersPrizeTotal = 0;
	
	for (int k = 0; k < partition; k++) {
		int currentResult = resultsArray[id + k];
		
		int prizeValue = 0;
		switch(currentResult) {
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
		
		// First machine to process
		if (previousMachinesResult == -2) {
			previousMachinesResult = currentResult;
			currentSameWinnerType = currentResult;
		}
		
		// First Scenario: All machines are the same winner
		if (areAllMachinesSameWinner && currentResult != currentSameWinnerType) {
			areAllMachinesSameWinner = 0;
		}
		
		// Second Scenario: All machines are some winner, but not all the same
		if (areAllMachinesAWinner && currentResult == -1) {
			areAllMachinesAWinner = 0;
		}
		else if (areAllMachinesAWinner) { // Keep totaling prize value
			allDifferentWinnersPrizeTotal += prizeValue * 2;
		}
		
		// Third Scenario: Not all machines are a winner.
		notAllWinnersPrizeTotal += prizeValue;
	}
	
	int prizeValue = 0;
	switch(currentSameWinnerType) {
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
	
	if (areAllMachinesSameWinner) {
		int factorial = 1;
		for (int k = partition; k > 1; k++) factorial *= k;
		pooledPrizeResults[y * 500 + x] = prizeValue * factorial;
	}
	else if (areAllMachinesAWinner) {
		pooledPrizeResults[y * 500 + x] = allDifferentWinnersPrizeTotal;
	}
	else { // Some or no winners
		pooledPrizeResults[y * 500 + x] = notAllWinnersPrizeTotal / partition;
	}
}

void PlayGame(int isDebug) {
	GpuTimer timer;
	
	// At least my hardware cannot handle one million machines, based upon our implementation
	int currentMoney = 100000;
	
	while(1) {
		if (currentMoney < 1000) {
			printf("You do not have sufficient funds. Thanks for playing Turbo Slots!");
			printf("Current Money: %i (Each slot machine is one money).\n", currentMoney);
			break;
		}
		
		printf("Current Money: %i\n", currentMoney);
		
		int numSlotMachines = GetNumberOfMachinesInput(playerInput, currentMoney);
		
		currentMoney -= numSlotMachines;
		
		int MACHINES_BY_FIVE_HUNDREDS = numSlotMachines / 500;
		
		// The player pressing Enter for the above input is read in the next input handling as ' ' for some reason...
		scanf("%c", playerInput);
		
		
		// Player Input for Pooling Machines or not
		// 		- We ran into some errors occuring on the second kernel for this and did not have time to work through them...
		//int isUserPoolingMachines = GetIsUserPoolingMachinesInput();
		int isUserPoolingMachines = 0;
		printf("Pooling Turbo Slot Machines is down for maintenance. So we have chosen non-pooling automatically for you.\nWe are sorry for the inconvenience.\n\n");
		printf("Please press enter to spin your machines!");
		scanf("%c", playerInput);

		// Backend RNG Setup
		const unsigned int blocks = MACHINES_BY_FIVE_HUNDREDS;
		const unsigned int threads = 500;
		const unsigned int totalThreads = MACHINES_BY_FIVE_HUNDREDS * 500;
		
		curandState *devStates;
		
		HANDLE_ERROR(cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState)));
		

		// Initialize Host and Device arrays
		if (isDebug) { printf("Size: %i \n", totalThreads); }
		unsigned int numbytes = totalThreads * sizeof(unsigned int);
		if (isDebug) { printf("Numbytes: %i \n", numbytes); }
		unsigned int *resultsArray = 0;
		unsigned int *pooledPrizeResults = 0;
		unsigned int *dev_resultsArray = 0;
		unsigned int *dev_pooledPrizeResults = 0;
		
		resultsArray = (unsigned int *) malloc(numbytes);
		HANDLE_ERROR(cudaMalloc((void**)&dev_resultsArray, numbytes));
		HANDLE_ERROR(cudaMemset(dev_resultsArray, 0, totalThreads * sizeof(unsigned int)));
		
		time_t seconds = time(NULL);
		
		setupRNG_kernel<<<blocks, threads>>>(devStates, seconds);
		
		int partition = 1;
		
		// Execute respective kernel
		if (isUserPoolingMachines) {
			partition = GetPartitionInput(playerInput);
			
			int numBytes = totalThreads / partition * sizeof(unsigned int);
			pooledPrizeResults = (unsigned int *) malloc(numBytes);
			HANDLE_ERROR(cudaMalloc((void**)&dev_pooledPrizeResults, numBytes));
			HANDLE_ERROR(cudaMemset(dev_pooledPrizeResults, 0, numBytes));
			
			timer.Start();
			PoolingIndividualMachinesKernel<<<blocks, threads>>>(devStates, dev_resultsArray, MACHINES_BY_FIVE_HUNDREDS);
			
			PoolingProcessingKernel<<<blocks, threads>>>(dev_resultsArray, dev_pooledPrizeResults, partition, totalThreads);
			timer.Stop();
		}
		else {
			timer.Start();
			NonPoolingKernel<<<blocks, threads>>>(devStates, dev_resultsArray, MACHINES_BY_FIVE_HUNDREDS);
			timer.Stop();
		}
		
		if (isUserPoolingMachines) {
			HANDLE_ERROR(cudaMemcpy(pooledPrizeResults, dev_pooledPrizeResults, totalThreads / partition * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		}
		else {
			HANDLE_ERROR(cudaMemcpy(resultsArray, dev_resultsArray, totalThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		}
		
		// Sequential
		if (isDebug) {			
			// Initialize Variables
			unsigned int seqNumBytes = totalThreads * sizeof(unsigned int);
			unsigned int* seqResultsArray = (unsigned int*)malloc(seqNumBytes);

			// Execute respective sequential implementation
			if (isUserPoolingMachines) {
				// TODO: We ran out of time and were not able to implement this.
			}
			else { // Non-pooling
				unsigned int values[3];
				auto starting = std::chrono::high_resolution_clock::now();

				for (int i = 0; i < totalThreads; i++)
				{
					// Generate pseudo-random unsigned ints
					values[0] = rand() % 10;
					values[1] = rand() % 10;
					values[2] = rand() % 10;

					// Assign the appropriate symbol based on percentage it should appear
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
						switch (values[0]) {
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

					seqResultsArray[i] = prizeValue;
				}

				auto elapsed = std::chrono::high_resolution_clock::now() - starting;
				long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

				printf("The sequential machines took %I64d microseconds\n", microseconds);
				printf("\n");
			}
			
			
			// OUTPUT RESULTS
			int gamesWon = 0;
			int gamesLost = 0;
			int zeroWinners = 0;
			int oneWinners = 0;
			int twoWinners = 0;
			int totalWinnings = 0;
			for (int k = 0; k < totalThreads; k++) {
				//printf("%i ", seqResultsArray[k]);
				if (seqResultsArray[k] > 0) gamesWon++;
				if (seqResultsArray[k] == 0) gamesLost++;
				if (seqResultsArray[k] == 3) zeroWinners++;
				if (seqResultsArray[k] == 10) oneWinners++;
				if (seqResultsArray[k] == 50) twoWinners++;
				totalWinnings += seqResultsArray[k];
			}
			printf("\n");
			printf("(Seq) Number of machines that were winners: %i\n", gamesWon);
			printf("(Seq) Number of machines that were losers: %i\n", gamesLost);
			printf("(Seq) Number of zero winners: %i\n", zeroWinners);
			printf("(Seq) Number of one winners: %i\n", oneWinners);
			printf("(Seq) Number of two winners: %i\n", twoWinners);
			printf("(Seq) Total winnings: %i\n", totalWinnings);
			
			
			 // CLEAN-UP
			free(seqResultsArray);
			
		}
		
		
		// OUTPUT RESULTS
		if (isUserPoolingMachines) {
			
		}
		else {
			int gamesWon = 0;
			int gamesLost = 0;
			int zeroWinners = 0;
			int oneWinners = 0;
			int twoWinners = 0;
			int totalWinnings = 0;
			for (int k = 0; k < totalThreads; k++) {
				//printf("%i ", resultsArray[k]);
				if (resultsArray[k] > 0) gamesWon++;
				if (resultsArray[k] == 0) gamesLost++;
				if (resultsArray[k] == 3) zeroWinners++;
				if (resultsArray[k] == 10) oneWinners++;
				if (resultsArray[k] == 50) twoWinners++;
				totalWinnings += resultsArray[k];
			}
			printf("\n");
			printf("Number of machines that were winners: %i\n", gamesWon);
			printf("Number of machines that were losers: %i\n", gamesLost);
			printf("Number of zero winners: %i\n", zeroWinners);
			printf("Number of one winners: %i\n", oneWinners);
			printf("Number of two winners: %i\n", twoWinners);
			printf("Total winnings: %i\n", totalWinnings);
			
			currentMoney += totalWinnings;
			
			if (isDebug) { printf("The machines took %g ms\n", timer.Elapsed()); }
			
			printf("\n");
		}
		
		// CLEAN-UP
		free(resultsArray);
		HANDLE_ERROR(cudaFree(dev_resultsArray));
		HANDLE_ERROR(cudaFree(devStates));
		if (isUserPoolingMachines) {
			free(pooledPrizeResults);
			HANDLE_ERROR(cudaFree(dev_pooledPrizeResults));
		}
		
		
		int doesUserWantToPlayAgain = GetUserPlayAgainInput();
		if (doesUserWantToPlayAgain == 0) {
			printf("\n");
			printf("Thanks for playing Turbo Slots!\n\n");
			printf("You ended your session with %i money!", currentMoney);
			return;
		}
		else { printf("\n"); }
	}
}

int main(int argc, char *argv[]) {
	int isDebug = 0;
	if (argc > 2) {
		printf("This program accepts only zero arguments. Please run again without any arguments\n");
		return 0;
	}
	else if (argc == 2 && strcmp(argv[1], "debug") == 0) {
		isDebug = 1;
	}
	else if (argc == 2) { 
		printf("This program accepts only zero arguments. Please run again without any arguments\n");
		return 0;
	}
	
	if (isDebug) {
		int nDevices;

		cudaGetDeviceCount(&nDevices);
		for (int i = 0; i < nDevices; i++) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			printf("\n");
			printf("Device Number: %d\n", i);
			printf("  Device name: %s\n", prop.name);
			printf("  Memory Clock Rate (KHz): %d\n",
				   prop.memoryClockRate);
			printf("  Memory Bus Width (bits): %d\n",
				   prop.memoryBusWidth);
			printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
				   2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		}
	}
	
	PrintEmptyLines(25);
	
	// Game intro and explanation
	printf("Welcome to Turbo Gambling!\n\n");
	printf("Most of our services are under construction. The only game available is Turbo Slot Machines\n");
	
	DisplayRules();

	
	// START GAME (LOOP)
	PlayGame(isDebug);

	PrintEmptyLines(2);

	return 0;
}