#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

__global__ void extreme(float** inputs, int sizei, int sizej, float** outputs) {
	int myrank = threadIdx.x;
	int myi = myrank / sizei;
	int myj = myrank % sizej;
	float mynum = inputs[myi][myj];
	bool ismax = true; // determine if we're larger than surrounding cells
	if(myi > 0) {
		if(inputs[myi-1][myj] >= mynum) {ismax = false;}
		if(myj > 0) {
			if(inputs[myi-1][myj-1] >= mynum) {ismax = false;}
			if(myj < sizej-1) {
				if(inputs[myi-1][myj+1] >= mynum) {ismax = false;}
			}
		}
	}
	if(myi < sizei-1) {
		if(inputs[myi+1][myj] >= mynum) {ismax = false;}
		if(myj > 0) {
			if(inputs[myi+1][myj-1] >= mynum) {ismax = false;}
			if(myj < sizej-1) {
				if(inputs[myi+1][myj+1] >= mynum) {ismax = false;}
			}
		}
	}
	if(myj > 0) {
		if(inputs[myi][myj-1] >= mynum) {ismax = false;}
	}
	if(myj < sizej - 1) {
		if(inputs[myi][myj+1] >= mynum) {ismax = false;}
	}
	if(ismax) {
		outputs[myrank][0] = myi;
		outputs[myrank][1] = myj;
		printf("found a match! rank %f", myrank);
	}
	__syncthreads();
}

int main() {
	string filein = "D:\\Documents\\Chapman\\CPSC445\\assignment05\\input.csv";
	string fileout = "output.csv";
	string line = "";
	vector<vector<float>> datavec;

	try {
		ifstream instream(filein);
		if(!instream.good()) {
			throw invalid_argument("File does not exist");
		}
		if(instream.is_open()) {
			while(getline(instream, line)) {
				istringstream sstream(line);
				string cell = "";
				vector<float> oneline;
				while(getline(sstream, cell, ',')) {
					oneline.push_back(stof(cell));
				}
				datavec.push_back(oneline);
			}
		}
		instream.close();
	}
	catch(exception& e) {
		cout << "Invalid input" << endl;
	}

	int sizei = datavec.size();
	int sizej = datavec[0].size();

	float** data = new float*[sizei];
	for(int i = 0; i < sizei; ++i) {
		data[i] = new float[sizej];
		for(int j = 0; j < sizej; ++j) {
			data[i][j] = datavec[i][j];
		}
	}

	float** inputs;
	cudaMalloc((void**)&inputs, sizei * sizej * sizeof(float));
	cudaMemcpy(inputs, data, sizei * sizej * sizeof(float), cudaMemcpyHostToDevice);

	float** placeholder = new float*[sizei * sizej];
	for(int i = 0; i < sizei * sizej; ++i) {
		placeholder[i] = new float[2];
		placeholder[i][0] = -1.0;
		placeholder[i][1] = -1.0;
	}

	float** outputs;
	cudaMalloc((void**)&outputs, sizei * sizej * sizeof(float));
	cudaMemcpy(outputs, placeholder, sizei * sizej * sizeof(float), cudaMemcpyHostToDevice);

	extreme<<<1,sizei*sizej>>>(inputs, sizei, sizej, outputs);
	cudaDeviceSynchronize();

	cudaMemcpy(placeholder, outputs, sizei * sizej * sizeof(float), cudaMemcpyDeviceToHost);

	ofstream outstream(fileout);
	for(int i = 0; i < sizei * sizej; ++i) {
		cout << "outputs here are " << placeholder[i][0] << "," << placeholder[i][1] << endl;
		if(placeholder[i][0] != -1.0 && placeholder[i][1] != -1.0) {
			outstream << placeholder[i][0] << "," << placeholder[i][1] << endl;
		}
	}
	outstream.close();

	cudaFree(inputs);
	cudaFree(outputs);
	free(data);
	free(placeholder);
	return 0;
}
