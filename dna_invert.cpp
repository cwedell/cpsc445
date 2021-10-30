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

  vector<char> myoutput;
  for(int i = 0; i < sizeeach; ++i) {
    char mychar = mydnachar[i];
    switch(mychar) {
      case 'A':
        myoutput.push_back('T');
        break;
      case 'T':
        myoutput.push_back('A');
        break;
      case 'G':
        myoutput.push_back('C');
        break;
      case 'C':
        myoutput.push_back('G');
        break;
      default:
        myoutput.push_back('\0');
        break;
    }
  }

  char output[sizeeach * p];
  check_error(MPI_Gather(myoutput.data(), sizeeach, MPI_CHAR, output, sizeeach, MPI_CHAR, 0, MPI_COMM_WORLD));

  if(rank == 0) {
    // print file out
    ofstream outstream(fileout);
    for(int i = 0; i < sizeeach * p; ++i) {
      if(output[i] == 'A' || output[i] == 'T' || output[i] == 'G' || output[i] == 'C') {
        string chartostr(1, output[i]);
        outstream << chartostr;
      }
    }
    outstream << endl;
    outstream.close();
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}
