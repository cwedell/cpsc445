#include "Grid.h"
#include <fstream>
#include <iostream>
// #include <mutex>
// #include <thread>
#include <stdexcept>
#include <vector>

using namespace std;

/*
mutex job_list_mtx;

void task(vector<int> myjob, Grid& curr, Grid& next) {
  next.setCell(myjob[0], myjob[1], curr.nextState(myjob[0], myjob[1]));
}

void pool(vector<vector<int>>& joblist, Grid& curr, Grid& next) {
  bool task_todo = false;
  vector<int> nextjob;
  while(!joblist.empty()) {
    job_list_mtx.lock();
    if(!joblist.empty()) { // second check to ensure list is not empty
      nextjob = joblist.pop_back();
      task_todo = true;
    }
    job_list_mtx.unlock();
    if(task_todo) {
      task(nextjob, curr, next);
    }
  }
}
*/

int main(int argc, char** argv) {
  string filein;
  string fileout;
  int numsteps;
  int numthreads;
  try {
    filein = argv[1];
    fileout = argv[2];
    numsteps = stoi(argv[3]);
    numthreads = stoi(argv[4]);
  } catch(exception& e) {
    cout << "Invalid arguments";
    return 1;
  }

  // read file
  int rowCount = 0;
  int colCount = 0;
  string line = "";
  vector<vector<int>> vals;
  try {
    ifstream instream (filein);
    if(!instream.good()) { // check that a valid file was given
      throw invalid_argument("File does not exist");
    }
    if(instream.is_open()) {
      while(getline(instream, line)) {
        string c = "";
        vector<int> row;
        for(int i = 0; i < line.size(); ++i) {
          c = line[i];
          if(stoi(c) != 0 && stoi(c) != 1) { // check that only 0 and 1 appear in file
            throw invalid_argument("Invalid input");
          }
          row.push_back(stoi(c));
          if(i == 0) { // get numcols from first line of file
            colCount = line.size();
          }
        }
        vals.push_back(row);
        ++rowCount;
      }
    }
    instream.close();
  } catch(exception& e) {
    cout << "Invalid input";
  }

  int numrows = rowCount;
  int numcols = colCount;

  Grid currentgrid(numrows, numcols, vals);
  Grid nextgrid(numrows, numcols, vals);

  /*
  vector<vector<int>> cells_todo;
  for(int i = 0; i < numrows; ++i) {
    for(int j = 0; j < numcols; ++j) {
      cells_todo.push_back(vector<int>{i, j});
    }
  }

  vector<thread*> threads;

  for(int i = 0; i < numthreads; ++i) {
    threads.push_back(pthread_create(&threads[i], NULL, pool(cells_todo, currentgrid, nextgrid)));
  }

  for(int i = 0; i < numthreads; ++i) {
    thread& t = *threads[i];
    t.join();
    delete threads[i];
  }

  threads.resize(0);
  */

  // play the game! calculate a cell value for each cell in each step
  for(int step = 0; step < numsteps; ++step) {
    for(int i = 0; i < numrows; ++i) {
      for(int j = 0; j < numcols; ++j) {
        nextgrid.setCell(i, j, currentgrid.nextState(i, j));
      }
    }
    currentgrid = nextgrid;
  }

  // print final grid to file
  ofstream outstream (fileout);
  for(int i = 0; i < numrows; ++i) {
    for(int j = 0; j < numcols; ++j) {
      outstream << currentgrid.getCell(i, j);
    }
    outstream << endl;
  }
  outstream.close();
}
