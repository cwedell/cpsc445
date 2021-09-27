#include "Grid.h"
#include <stdexcept>
#include <vector>

using namespace std;

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
