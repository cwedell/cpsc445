#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

__global__ void sqrtcalc(float* inputs) {
	float mynum = inputs[threadIdx.x];
	mynum = sqrt(mynum);
	inputs[threadIdx.x] = mynum;
	__syncthreads();
}

int main() {
	string filein = "input.csv";
	string fileout = "output.csv";
	vector<float> nums;
	string line = "";

	try {
		ifstream instream(filein);
		if(!instream.good()) {
			throw invalid_argument("File does not exist");
		}
		if(instream.is_open()) {
			while(getline(instream, line)) {
				nums.push_back(stof(line));
			}
		}
		instream.close();
	}
	catch(exception& e) {
		cout << "Invalid input" << endl;
	}

	int size = nums.size();
	float* sqrts = &nums[0];
	float* inputs;
	cudaMalloc((void**)&inputs, size * sizeof(float));
	cudaMemcpy(inputs, sqrts, size * sizeof(float), cudaMemcpyHostToDevice);

	sqrtcalc<<<1, size>>>(inputs);
	cudaDeviceSynchronize();

	float* outputs = new float[size];
	cudaMemcpy(outputs, inputs, size * sizeof(float), cudaMemcpyDeviceToHost);

	ofstream outstream(fileout);
	for(int i = 0; i < size; ++i) {
		outstream << outputs[i] << endl;
    cout << "sqrt of " << nums[i] << " = " << outputs[i] << endl;
	}
	outstream.close();

	cudaFree(inputs);
	free(outputs);
	return 0;
}
