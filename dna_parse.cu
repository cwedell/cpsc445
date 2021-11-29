#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

__global__ void charparse(char* dnainput, int* tripletsinput, int size) {
	int mytriplet = tripletsinput[threadIdx.x];
	int tripletcount = 0;
	for(int i = 0; i < size; ++i) {
		if(i % 3 == 2) { // first runs for index=2
			char arrtocheck[] = {dnainput[i - 2], dnainput[i - 1], dnainput[i]};
			int inttocheck = 0;

      // convert char triplet to int representation, since map is unavailable
			if(arrtocheck[0] == 'A') {inttocheck += 100;}
			if(arrtocheck[0] == 'T') {inttocheck += 200;}
			if(arrtocheck[0] == 'G') {inttocheck += 300;}
			if(arrtocheck[0] == 'C') {inttocheck += 400;}
			if(arrtocheck[1] == 'A') {inttocheck += 10;}
			if(arrtocheck[1] == 'T') {inttocheck += 20;}
			if(arrtocheck[1] == 'G') {inttocheck += 30;}
			if(arrtocheck[1] == 'C') {inttocheck += 40;}
			if(arrtocheck[2] == 'A') {inttocheck += 1;}
			if(arrtocheck[2] == 'T') {inttocheck += 2;}
			if(arrtocheck[2] == 'G') {inttocheck += 3;}
			if(arrtocheck[2] == 'C') {inttocheck += 4;}

			if(inttocheck == mytriplet) {
				++tripletcount;
			}
		}
	}
	tripletsinput[threadIdx.x] = tripletcount;
	__syncthreads();
}

int main() {
	string filein = "dna.txt";
	string fileout = "output.txt";
	string dna = "";
	string line = "";
	map<string, int> triplettable = { // integer representations of each possible triplet
	{"AAA", 111}, {"AAT", 112}, {"AAG", 113}, {"AAC", 114},
	{"ATA", 121}, {"ATT", 122}, {"ATG", 123}, {"ATC", 124},
	{"AGA", 131}, {"AGT", 132}, {"AGG", 133}, {"AGC", 134},
	{"ACA", 141}, {"ACT", 142}, {"ACG", 143}, {"ACC", 144},
	{"TAA", 211}, {"TAT", 212}, {"TAG", 213}, {"TAC", 214},
	{"TTA", 221}, {"TTT", 222}, {"TTG", 223}, {"TTC", 224},
	{"TGA", 231}, {"TGT", 232}, {"TGG", 233}, {"TGC", 234},
	{"TCA", 241}, {"TCT", 242}, {"TCG", 243}, {"TCC", 244},
	{"GAA", 311}, {"GAT", 312}, {"GAG", 313}, {"GAC", 314},
	{"GTA", 321}, {"GTT", 322}, {"GTG", 323}, {"GTC", 324},
	{"GGA", 331}, {"GGT", 332}, {"GGG", 333}, {"GGC", 334},
	{"GCA", 341}, {"GCT", 342}, {"GCG", 343}, {"GCC", 344},
	{"CAA", 411}, {"CAT", 412}, {"CAG", 413}, {"CAC", 414},
	{"CTA", 421}, {"CTT", 422}, {"CTG", 423}, {"CTC", 424},
	{"CGA", 431}, {"CGT", 432}, {"CGG", 433}, {"CGC", 434},
	{"CCA", 441}, {"CCT", 442}, {"CCG", 443}, {"CCC", 444}
	};

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

	// this vector exists only to ensure duplicate triplets are not included
	vector<string> triplets;
	for(int i = 0; i < dna.length() - 1; ++i) {
		if(i % 3 == 2) { // first runs for index=2
			char tripletarr[] = {dnachar[i - 2], dnachar[i - 1], dnachar[i], '\0'};
			string tripletstr = tripletarr;
			if(find(triplets.begin(), triplets.end(), tripletstr) == triplets.end()) { // true if duplicate not found
				triplets.push_back(tripletstr);
			}
		}
	}
	int vecsize = triplets.size();

	int* tripletsarr = new int[vecsize];
	for(int i = 0; i < vecsize; ++i) {
		tripletsarr[i] = triplettable[triplets[i]]; // convert vector of triplets to int representation
	}

  // send int representation of triplets to device
	int* tripletsinput;
	cudaMalloc((void**)&tripletsinput, vecsize * sizeof(int));
	cudaMemcpy(tripletsinput, tripletsarr, vecsize * sizeof(int), cudaMemcpyHostToDevice);

  // one thread per triplet found
	charparse<<<1, vecsize>>>(dnainput, tripletsinput, size);
	cudaDeviceSynchronize();

  // get triplet counts from device
	int* tripletsoutput = new int[vecsize];
	cudaMemcpy(tripletsoutput, tripletsinput, vecsize * sizeof(int), cudaMemcpyDeviceToHost);

	ofstream outstream(fileout);
	for(int i = 0; i < vecsize; ++i) {
		outstream << triplets[i] << " " << tripletsoutput[i] << endl;
	}
	outstream.close();

	cudaFree(dnainput);
	cudaFree(tripletsinput);
  free(dnachar);
  free(tripletsarr);
	free(tripletsoutput);
	return 0;
}
