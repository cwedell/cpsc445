#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <vector>

using namespace std;

void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

int main (int argc, char *argv[]) {
  int rank;
  int p;

  string filein = "dna.txt";
  string fileout = "output.txt";
  string dna = "";
  string line = "";
  char letters[] = {'A', 'T', 'G', 'C'};
  vector<int> output;
  output.push_back(0);
  output.push_back(0);
  output.push_back(0);
  output.push_back(0);

  // Initialized MPI
  check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  cout << "Starting process " << rank << "/" << "p\n";

  // read file in
  if(rank == 0) {
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
    } catch(exception& e) {
      cout << "Invalid input" << endl;
    }
  }

  char dnachar[dna.length() + 1];
  strcpy(dnachar, dna.c_str());
  double size = dna.length();
  int sizeeach = (int) (ceil(size / p)); // size of array to be handled by each process
  check_error(MPI_Bcast(&sizeeach, 1, MPI_INT, 0, MPI_COMM_WORLD));
  char mydnachar[sizeeach];
  check_error(MPI_Scatter(dnachar, sizeeach, MPI_CHAR, mydnachar, sizeeach, MPI_CHAR, 0, MPI_COMM_WORLD));

  int countA = 0, countT = 0, countG = 0, countC = 0;
  for(int i = 0; i < sizeeach; ++i) {
    char mychar = mydnachar[i];
    switch(mychar) {
      case 'A':
        ++countA;
        break;
      case 'T':
        ++countT;
        break;
      case 'G':
        ++countG;
        break;
      case 'C':
        ++countC;
        break;
    }
  }
  int myoutput[] = {countA, countT, countG, countC};

  int alloutputs[p * 4];
  check_error(MPI_Gather(myoutput, 4, MPI_INT, alloutputs, 4, MPI_INT, 0, MPI_COMM_WORLD));

  if(rank == 0) {
    for(int i = 0; i < p; ++i) {
      for(int j = 0; j < 4; ++j) {
        output[j] += alloutputs[i * 4 + j];
      }
    }
    // print file out
    ofstream outstream(fileout);
    for(int i = 0; i < 4; ++i) {
      string chartostr(1, letters[i]);
      outstream << chartostr << " ";
      outstream << output[i] << endl;
    }
    outstream.close();
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}
