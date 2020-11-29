#ifndef __FINALPROJECT_HELPERS_H__
#define __FINALPROJECT_HELPERS_H__

void PrintEmptyLines(int numLines) {
	for (int k = 0; k < numLines; k++) {
		printf("\n");
	}
}

void DisplayRules() {
	printf("\nTURBO SLOTS RULES\n");
	printf("------------------------\n");
	printf("\n");
	printf("At its core, Turbo Slots is very similar to any standard slot machine.\n");
	printf("There are two major differences.\n");
	printf("	(1) The player spins 1000s of machines all at once.\n");
	printf("	(2) The player has the option to pool their machines into larger Combo Slots.\n");
	printf("		These Combo Machines concatentate their results together making it harder to win.\n");
	printf("		But when you do, you are rewarded factorially!\n");
	printf("\n");
	
	printf("Examples of pooled machine results");
	printf("	Partial Winner:\n");
	printf("		|201|012|111|\n");
	printf("			These three machines pooled together would be a partial winner.\n");
	printf("			The third machine is still considered a winner on its own, but\n");
	printf("			the prize is divided by the total number of machines in the pool.\n");
	printf("	Turbo Winner:\n");
	printf("		|222|000|111|\n");
	printf("			When all pooled machines are winners of different symbols, then\n");
	printf("			all prizes are doubled!\n");
	printf("\n");
	printf("	Turbo Jackpot:\n");
	printf("		|111|111|111|\n");
	printf("			This pool of machines is what creates the incredible excitement\n");
	printf("			around Turbo Slots! When all pooled machines come up the same symbol\n");
	printf("			it is a TURBO JACKPOT! You win that symbols prize multiplied by the\n");
	printf("			factorial of the total number of machines in the pool! So here it would\n");
	printf("			be 3*2*1*\"111\"s prize value!\n");
	printf("\n");
}

int GetNumberOfMachinesInput(char* playerInput, int currentMoney) {
	int numSlotMachines = 1;
	while(numSlotMachines % 1000 != 0 || numSlotMachines > currentMoney) {
		printf("How many Turbo Slot Machines would you like to play (by 1000s): ");
		scanf("%s", playerInput);
		numSlotMachines = strtol(playerInput, (char **)NULL, 10);
		if (numSlotMachines % 1000 != 0) printf("%d is not a valid number of machines, try a multiple of 1000.\n", numSlotMachines);
		if (numSlotMachines > currentMoney) printf("You only have %i money, please enter a number of machines less than or equal to %i.\n", currentMoney, currentMoney);
	}
	
	printf("Please wait while we prepare your %d machines.\n\n", numSlotMachines);
	return numSlotMachines;
}

int GetUserPoolingMachinesInput() {
	char answer = ' ';
	int error = 0;
	while(answer != 'y' && answer != 'n' && answer != 'Y' && answer != 'N') {
		if (error) printf("\n%c is not a valid answer. Try again.\n", answer);
		printf("Would you like to pool your machines? (Y/N): ");
		scanf("%c", &answer);
		error++;
	}
	
	if (answer == 'y' || answer == 'Y') return 1;
	
	return 0;
}

int GetUserPlayAgainInput() {
	char answer = ' ';
	int error = 0;
	while(answer != 'y' && answer != 'n' && answer != 'Y' && answer != 'N') {
		if (error) printf("\n%c is not a valid answer. Try again.\n", answer);
		printf("Would you like to keep playing? (Y/N): ");
		scanf("%c", &answer);
		error++;
	}
	
	if (answer == 'y' || answer == 'Y') return 1;
	
	return 0;
}

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#endif // __FINALPROJECT_HELPERS_H__

/*
	printf("   __.--~~.,-.__		\n");
	printf("   `~-._.-(`-.__`-.		\n");
	printf("           \\    `~~`	\n");
	printf("      .--./ \\			\n");
	printf("     /#   \\  \\.--.		\n");
	printf("     \\    /  /#   \\		\n");
	printf("      '--'   \\    /		\n");
	printf("              '--'		\n");
	*/
	
	// PrintEmptyLines(5);
	
	/*
	printf("                _.----.	    \n");
	printf("              ,'       `\\	\n");
	printf("             /           \\	\n");
	printf("            /            |	\n");
	printf("           /             ;	\n");
	printf("        _,'             /	\n");
	printf("       (_             /		\n");
	printf("         '-._        /		\n");
	printf("            //-._   (		\n");
	printf("           //    `-._)		\n");
	printf("         (_)        		\n");
	*/

/*
// The following is using cuRAND from the host
	size_t n = 100;
    //float *devData, *hostData;
    unsigned int *devData, *hostData;
	curandGenerator_t gen;
	
	// Allocate n floats on host
    //hostData = (float *)calloc(n, sizeof(float));
    hostData = (unsigned int *)calloc(n, sizeof(unsigned int));

    // Allocate n floats on device
    //HANDLE_ERROR(cudaMalloc((void **)&devData, n*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&devData, n*sizeof(unsigned int)));
	
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
	
	// Generate n floats on device
    //CURAND_CALL(curandGenerateUniform(gen, devData, n));
    CURAND_CALL(curandGenerate(gen, devData, n));

    // Copy device memory to host
    //HANDLE_ERROR(cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(hostData, devData, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Show result
    for(int k = 0; k < n; k++) {
        //printf("%1.4f ", hostData[k]);
        printf("%i ", hostData[k]);
    }
    printf("\n");
	
	
	CURAND_CALL(curandDestroyGenerator(gen));
	HANDLE_ERROR(cudaFree(devData));
    free(hostData);
*/