#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
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

  // cannot broadcast strings, only char arrays
  char dnachar[dna.length() + 1];
  strcpy(dnachar, dna.c_str());

  // this vector exists only to ensure duplicate triplets are not included
  vector<string> triplets;
  if(rank == 0) {
    for(int i = 0; i < sizeof(dnachar) - 1; ++i) {
      if(i % 3 == 2) { // first runs for index=2
        char tripletarr[] = {dnachar[i - 2], dnachar[i - 1], dnachar[i]};
        string tripletstr = tripletarr;
        if(find(triplets.begin(), triplets.end(), tripletstr) == triplets.end()) { // true if duplicate not found
          triplets.push_back(tripletstr);
        }
      }
    }
    for(int i = 0; i < triplets.size(); ++i) {
      int ranktosend = (i % (p - 1)) + 1; // assigns index=0 to rank=1 and loops
      char arrtosend[3];
      strcpy(arrtosend, triplets[i].c_str());
      check_error(MPI_Send(arrtosend, 3, MPI_CHAR, ranktosend, 0, MPI_COMM_WORLD)); // tag 0
    }
  }

  // also pass size, for proper iteration
  int arrsize = dna.length() + 1;
  check_error(MPI_Bcast(&arrsize, 1, MPI_INT, 0, MPI_COMM_WORLD));
  check_error(MPI_Bcast(dnachar, arrsize, MPI_CHAR, 0, MPI_COMM_WORLD));

  int vecsize = triplets.size();
  check_error(MPI_Bcast(&vecsize, 1, MPI_INT, 0, MPI_COMM_WORLD));

  if(rank != 0) {
    for(int i = 0; i < vecsize; ++i) {
      if(rank == (i % (p - 1)) + 1) { // assigns index=0 to rank=1 and loops
        MPI_Status status[2];
        char arrtorecv[3];
        check_error(MPI_Recv(arrtorecv, 3, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status[0])); // tag 0
        int tripletcount = 0;
        for(int j = 0; j < arrsize; ++j) {
          if(j % 3 == 2) { // first runs for index=2
            char arrtocheck[] = {dnachar[j - 2], dnachar[j - 1], dnachar[j]};
            if(arrtorecv[0] == arrtocheck[0] && arrtorecv[1] == arrtocheck[1] && arrtorecv[2] == arrtocheck[2]) {
              ++tripletcount;
            }
          }
        }
        arrtorecv[3] = '\0'; // terminate array at desired point
        check_error(MPI_Send(arrtorecv, 3, MPI_CHAR, 0, 1, MPI_COMM_WORLD)); // tag 1
        check_error(MPI_Send(&tripletcount, 1, MPI_INT, 0, 2, MPI_COMM_WORLD)); // tag 2
      }
    }
  }

  map<string, int> output;
  if(rank == 0) {
    MPI_Status status[2];
    for(int i = 1; i < p; ++i) { // start receiving from process 1
      for(int j = 0; j < vecsize; ++j) {
        if(j % (p - 1) == (i - 1)) { // assigns index=0 to rank=1 and loops
          char arrrecvd[3];
          check_error(MPI_Recv(arrrecvd, 3, MPI_CHAR, i, 1, MPI_COMM_WORLD, &status[0])); // tag 1
          int outputcount;
          check_error(MPI_Recv(&outputcount, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status[0])); // tag 2
          string outputstr = arrrecvd;
          output[outputstr] = outputcount;
        }
      }
    }
    // print file out
    ofstream outstream(fileout);
    map<string, int>::iterator it;
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
