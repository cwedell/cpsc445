#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>

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
  char letters[] = {'A', 'C', 'G', 'T'};
  map<char, int> output = {{'A', 0}, {'C', 0}, {'G', 0}, {'T', 0}};

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

  // cannot broadcast strings, only char arrays
  char dnachar[dna.length() + 1];
  strcpy(dnachar, dna.c_str());
  // also pass size, for proper iteration
  int size = dna.length() + 1;
  check_error(MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD));
  check_error(MPI_Bcast(dnachar, size, MPI_CHAR, 0, MPI_COMM_WORLD));

  if(rank != 0) {
    char mychar = letters[rank - 1]; // -1 so that rank=1 handles index=0
    int charcount = 0;
    for(int i = 0; i < size; ++i) {
      if(dnachar[i] == mychar) {
        ++charcount;
      }
    }
    check_error(MPI_Send(&charcount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD)); // tag 0
  }

  if(rank == 0) {
    int count;
    MPI_Status status[2];
    for(int i = 1; i < p; ++i) { // start receiving from process 1
      check_error(MPI_Recv(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status[0])); // tag 0
      output[letters[i - 1]] = count; // -1 since rank=1 handled index=0
    }
    // print file out
    ofstream outstream(fileout);
    map<char, int>::iterator it;
    for(it = output.begin(); it != output.end(); ++it) {
      outstream << it->first << " ";
      outstream << it->second << endl;
    }
    outstream.close();
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}
