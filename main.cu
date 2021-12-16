#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

/*
The solver function determines the area of a slice of the shape.
It takes the dimension (width) of the slice, the x and y coefficients, the constant, and the starting x.
Its assigned slice corresponds to its rank, so it increments x until reaching that point.
Then, it solves the function for y to find the positive and negative root. Multiplying the difference between the roots and the width gives
the area of the slice.
*/

__global__ void solver(double* outputs, double dim, double xcoeff, double ycoeff, double constant, double xstart) {
	int myrank = blockIdx.x * blockDim.x + threadIdx.x;
	double x = xstart + (dim * myrank);
	if(x < abs(xstart * 2)) {
		/*
		Given the function Ax^2 + By^2 + C = 0, we can solve for y.
		Ax^2 + By^2 = -C
		By^2 = -C - Ax^2
		y^2 = (-C - Ax^2)/B
		y = sqrt((-C - Ax^2)/B)
		This is the equivalent of the assignment to y1 - the absolute value is added to ensure the input to the sqrt is positive.
		*/
		double y1 = sqrt(abs((-constant - xcoeff*pow(x,2))/ycoeff));
		double y2 = -y1;
		double area = dim*(y1-y2);
		outputs[myrank] = area;
	}
	__syncthreads();
}

int main() {
	double dim = 0.001; // change this to change precision

	// sample input, representing the equation 2x^2 + 3y^2 - 5 = 0
	double xcoeff = 2;
	double ycoeff = 3;
	double constant = -5;

	double area = 0;
	double xstart = -sqrt(abs(constant/xcoeff)); // the leftmost differentiable x
	int width = floor(abs(xstart)*2/dim); // the total width of the shape

	double* placeholder = new double[width];
	double* outputs;
	cudaMalloc((void**)&outputs, width * sizeof(double));
	cudaMemcpy(outputs, placeholder, width * sizeof(double), cudaMemcpyHostToDevice);

	solver<<<ceil((float)width/1024),1024>>>(outputs, dim, xcoeff, ycoeff, constant, xstart);
	cudaDeviceSynchronize();

	cudaMemcpy(placeholder, outputs, width * sizeof(double), cudaMemcpyDeviceToHost);

	for(int i = 0; i < width; ++i) {
		area += placeholder[i];
	}

	cout << "The area of the shape defined by " << (int)xcoeff << "x^2 + " << (int)ycoeff << "y^2 + " << (int)constant << " = " << area << endl;

	cudaFree(outputs);
	free(placeholder);
	return 0;
}
