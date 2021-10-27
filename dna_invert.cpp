#include <cstring>
#include <fstream>
#include <iostream>
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
  char letters[] = {'A', 'C', 'G', 'T'};
  char oppletters[] = {'T', 'G', 'C', 'A'};

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

  // create char array for output, plus null terminator
  char outputarr[dna.length() - 1];
  outputarr[dna.length() - 1] = '\0';

  // also pass size, for proper iteration
  int size = dna.length() + 1;
  check_error(MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD));
  check_error(MPI_Bcast(dnachar, size, MPI_CHAR, 0, MPI_COMM_WORLD));

  if(rank != 0) {
    char mychar = letters[rank - 1]; // -1 so that rank=1 handles index=0
    // vector of indexes matching letter, will swap later
    vector<int> charmatch;
    for(int i = 0; i < size; ++i) {
      if(dnachar[i] == mychar) {
        charmatch.push_back(i);
      }
    }
    int vectorsize = charmatch.size();
    check_error(MPI_Send(&vectorsize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD)); // tag 0
    // vector.data() returns a pointer, so no reference necessary
    check_error(MPI_Send(charmatch.data(), vectorsize, MPI_INT, 0, 1, MPI_COMM_WORLD)); // tag 1
  }

  if(rank == 0) {
    int vecsize;
    vector<int> indexes;
    MPI_Status status[2];
    for(int i = 1; i < p; ++i) { // start receiving from process 1
      check_error(MPI_Recv(&vecsize, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status[0])); // tag 0
      indexes.resize(vecsize); // prepare vector to handle incoming receive
      check_error(MPI_Recv(indexes.data(), vecsize, MPI_INT, i, 1, MPI_COMM_WORLD, &status[0])); // tag 1
      while(!indexes.empty()) {
        int idx = indexes.back();
        // put the opposite letter to the matching index in our output char array
        outputarr[idx] = oppletters[i - 1]; // -1 since rank=1 handled index=0
        indexes.pop_back();
      }
    }
    string dnaout = outputarr;
    // print file out
    ofstream outstream(fileout);
    outstream << dnaout << endl;
    outstream.close();
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}
