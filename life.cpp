#include <fstream>
#include <iostream>
#include <math.h>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std;

class Grid {
public:
  Grid(int r, int c, vector<vector<int>> v);
  int getCell(int r, int c);
  void setCell(int r, int c, int v);
  int countNeighbors(int r, int c);
  int nextState(int r, int c);

private:
  int rows;
  int cols;
  vector<vector<int>> values;
};

Grid::Grid(int r, int c, vector<vector<int>> v) {
  rows = r;
  cols = c;
  for(int i = 0; i < rows; ++i) {
    vector<int> row;
    for(int j = 0; j < cols; ++j) {
      row.push_back(v[i][j]);
    }
    values.push_back(row);
  }
}

int Grid::getCell(int r, int c) {
  return values[r][c];
}

void Grid::setCell(int r, int c, int v) {
  values[r][c] = v;
}

int Grid::countNeighbors(int r, int c) {
  int count = 0;
  if(r > 0) { // count cells to the left
    count += getCell(r - 1, c);
    if(c > 0) {
      count += getCell(r - 1, c - 1);
    }
    if(c < cols - 1) {
      count += getCell(r - 1, c + 1);
    }
  }
  if(r < rows - 1) { // count cells to the right
    count += getCell(r + 1, c);
    if(c > 0) {
      count += getCell(r + 1, c - 1);
    }
    if(c < cols - 1) {
      count += getCell(r + 1, c + 1);
    }
  }
  if(c > 0) { // count cell directly above
    count += getCell(r, c - 1);
  }
  if(c < cols - 1) { // count cell directly below
    count += getCell(r, c + 1);
  }
  return count;
}

int Grid::nextState(int r, int c) {
  int current = getCell(r, c);
  int neighbors = countNeighbors(r, c);
  if(current == 1 && neighbors < 2) { // underpopulation
    return 0;
  } else if(current == 1 && neighbors < 4) { // survival
    return 1;
  } else if(current == 1 && neighbors > 3) { // overpopulation
    return 0;
  } else if(current == 0 && neighbors == 3) { // reproduction
    return 1;
  } else {
    return 0;
  }
}

class Task { // a row and column to calculate
public:
  int r, c;
  Task(int r, int c) {
    this->r = r;
    this->c = c;
  }
};

mutex tasks_mtx; // protects vector of tasks

void TaskDoer(vector<Task> &vect, Grid &curr, Grid &next, int numtasks) {
  bool task_todo = false;
  Task mytask(-1, -1);
  for(int i = 0; i < numtasks; ++i) { // only run to complete the number of tasks assigned
    tasks_mtx.lock();
    if(!vect.empty()) { // ensure list is not empty
      mytask = vect.back();
      vect.pop_back();
      task_todo = true;
    }
    tasks_mtx.unlock();
    if(task_todo) {
      // update the value in the next grid
      next.setCell(mytask.r, mytask.c, curr.nextState(mytask.r, mytask.c));
    }
    task_todo = false;
  }
}

vector<Task> tasks;

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
    cout << "Invalid arguments" << endl;
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
        string c;
        vector<int> row;
        for(int i = 0; i < line.size() - 1; ++i) { // -1 to strip newline from end of line
          c = line[i];
          if(c != "0" && c != "1") { // check that only 0 and 1 appear in file
            throw invalid_argument("Invalid input");
          }
          row.push_back(stoi(c));
          if(i == 0) { // get numcols from first line of file
            colCount = line.size() - 1;
          }
        }
        vals.push_back(row);
        ++rowCount;
      }
    }
    instream.close();
  } catch(exception& e) {
    cout << "Invalid input" << endl;
  }

  int numrows = rowCount;
  int numcols = colCount;

  Grid currentgrid(numrows, numcols, vals);
  Grid nextgrid(numrows, numcols, vals);

  // play the game!
  for(int step = 0; step < numsteps; ++step) {
    // first create a list of tasks to do: one for each cell
    for(int i = 0; i < numrows; ++i) {
      for(int j = 0; j < numcols; ++j) {
        tasks.push_back(Task(i, j));
      }
    }

    vector<thread*> threads;

    // assign a number of tasks to each thread, such that all threads run and the task vector is emptied
    int numtasks = int(ceil(float(tasks.size()) / float(numthreads)));

    for(int i = 0; i < numthreads; ++i) {
      threads.push_back(new thread([&,i](){
        TaskDoer(tasks, currentgrid, nextgrid, numtasks);
      }));
    }

    for(int i = 0; i < numthreads; ++i) {
      thread& t = *threads[i];
      t.join();
      delete threads[i];
    }

    threads.resize(0);

    // grab the next grid, fully updated, and assign it to be the current grid
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
