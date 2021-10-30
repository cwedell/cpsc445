#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
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
  map<string, int> output;

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
  }

  int vecsize = triplets.size();
  check_error(MPI_Bcast(&vecsize, 1, MPI_INT, 0, MPI_COMM_WORLD));

  int tripletsarr[vecsize];
  if(rank == 0) {
    for(int i = 0; i < vecsize; ++i) {
      tripletsarr[i] = triplettable[triplets[i]]; // convert vector of triplets to int representation
    }
  }

  check_error(MPI_Bcast(tripletsarr, vecsize, MPI_INT, 0, MPI_COMM_WORLD));

  double size = dna.length();
  int sizeeach = (int) (ceil(size / p)); // size of array to be handled by each process
  if(sizeeach % 3 != 0) { // ensure sizeeach is divisible by 3
    sizeeach += (3 - (sizeeach % 3));
  }
  check_error(MPI_Bcast(&sizeeach, 1, MPI_INT, 0, MPI_COMM_WORLD));
  char mydnachar[sizeeach];
  check_error(MPI_Scatter(dnachar, sizeeach, MPI_CHAR, mydnachar, sizeeach, MPI_CHAR, 0, MPI_COMM_WORLD));

  int myoutput[vecsize];
  for(int i = 0; i < vecsize; ++i) {
    int tripletcount = 0;
    for(int j = 0; j < sizeeach; ++j) {
      if(j % 3 == 2) { // first runs for index=2
        char arrtocheck[] = {mydnachar[j - 2], mydnachar[j - 1], mydnachar[j]};
        string strtocheck = arrtocheck;
        if(triplettable[strtocheck] == tripletsarr[i]) {
          ++tripletcount;
        }
      }
    }
    myoutput[i] = tripletcount;
  }

  int alloutput[p * vecsize];
  check_error(MPI_Gather(myoutput, vecsize, MPI_INT, alloutput, vecsize, MPI_INT, 0, MPI_COMM_WORLD));

  if(rank == 0) {
    map<string, int> output;
    for(int j = 0; j < vecsize; ++j) {
      output[triplets[j]] = 0; // initialize all map values to 0
    }
    for(int i = 0; i < p; ++i) {
      for(int j = 0; j < vecsize; ++j) {
        output[triplets[j]] += alloutput[i * vecsize + j];
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
