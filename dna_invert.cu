#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;

__global__ void charinvert(char* dnainput) {
	char* mychar = (char*)dnainput[threadIdx.x];
	if(mychar == (char*)'A') {
		dnainput[threadIdx.x] = 'T';
	} else if(mychar == (char*)'T') {
		dnainput[threadIdx.x] = 'A';
	} else if(mychar == (char*)'G') {
		dnainput[threadIdx.x] = 'C';
	} else if(mychar == (char*)'C') {
		dnainput[threadIdx.x] = 'G';
	} else {
		dnainput[threadIdx.x] = '\0';
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

	char* dnainput;
	cudaMalloc((void**)&dnainput, size * sizeof(char));

	cudaMemcpy(dnainput, dnachar, size * sizeof(char), cudaMemcpyHostToDevice);

	charinvert<<<1, size>>>(dnainput);
	cudaDeviceSynchronize();

	char* dnaout = new char[dna.length() + 1];
	cudaMemcpy(dnaout, dnainput, size * sizeof(char), cudaMemcpyDeviceToHost);

	ofstream outstream(fileout);
	for (int i = 0; i < size; ++i) {
		if (dnaout[i] == 'A' || dnaout[i] == 'T' || dnaout[i] == 'G' || dnaout[i] == 'C') {
			string chartostr(1, dnaout[i]);
			outstream << chartostr;
		}
	}
	outstream << endl;
	outstream.close();

	cudaFree(dnainput);
	free(dnachar);
	free(dnaout);
	return 0;
}
