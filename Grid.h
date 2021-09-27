#include <iostream>
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
