#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;

__global__ void charcount(char* dnainput, int* countA, int* countT, int* countG, int* countC) {
	char* mychar = (char*)dnainput[threadIdx.x];
  // atomicAdd() adds 1 to each variable and prevents race conditions
	if(mychar == (char*)'A') {
		atomicAdd(countA, 1);
	} else if(mychar == (char*)'T') {
		atomicAdd(countT, 1);
	} else if(mychar == (char*)'G') {
		atomicAdd(countG, 1);
	} else if(mychar == (char*)'C') {
		atomicAdd(countC, 1);
	}
	__syncthreads();
}

int main() {
	string filein = "dna.txt";
	string fileout = "output.txt";
	string dna = "";
	string line = "";

	try {
		ifstream instream(filein);
		if(!instream.good()) {
			throw invalid_argument("File does not exist");
		}
		if(instream.is_open()) {
			while(getline(instream, line)) {
				dna = line;
			}
		}
		instream.close();
	}
	catch(exception& e) {
		cout << "Invalid input" << endl;
	}

	char* dnachar = new char[dna.length() + 1];
	strcpy(dnachar, dna.c_str());
	int size = dna.length();

  // send char array to device
	char* dnainput;
	cudaMalloc((void**)&dnainput, size * sizeof(char));
	cudaMemcpy(dnainput, dnachar, size * sizeof(char), cudaMemcpyHostToDevice);

  // send counters to device
	int* countA;
	int* countT;
	int* countG;
	int* countC;
	cudaMalloc((void**)&countA, sizeof(int));
	cudaMalloc((void**)&countT, sizeof(int));
	cudaMalloc((void**)&countG, sizeof(int));
	cudaMalloc((void**)&countC, sizeof(int));
	cudaMemcpy(countA, 0, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(countT, 0, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(countG, 0, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(countC, 0, sizeof(int), cudaMemcpyHostToDevice);

  // one thread per character
	charcount<<<1, size>>>(dnainput, countA, countT, countG, countC);
	cudaDeviceSynchronize();

  // get counters from device
	int Acounted;
	int Tcounted;
	int Gcounted;
	int Ccounted;
	cudaMemcpy(&Acounted, countA, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&Tcounted, countT, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&Gcounted, countG, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&Ccounted, countC, sizeof(int), cudaMemcpyDeviceToHost);

	ofstream outstream(fileout);
	outstream << "A " << Acounted << endl;
	outstream << "T " << Tcounted << endl;
	outstream << "G " << Gcounted << endl;
	outstream << "C " << Ccounted << endl;
	outstream.close();

	cudaFree(dnainput);
	free(dnachar);
	return 0;
}
