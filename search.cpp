#include <bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std;

mutex results_mtx; // protects map of results

void TextSearch(vector<string> keys, vector<string> tosearch, map<string, int> &results) {
  for(int i = 0; i < keys.size(); ++i) {
    for(int j = 0; j < tosearch.size(); ++j) {
      int newmatches = 0;
      string tomatch = keys[i];
      string searchtext = tosearch[j];
      cout << "searching for " << tomatch << " in " << searchtext << endl;
      size_t pos = searchtext.find(tomatch, 0);
      while(pos != string::npos) {
        cout << "match found!" << endl;
        ++newmatches;
        pos = searchtext.find(tomatch, pos + 1);
      }
      results_mtx.lock();
      results[tomatch] += newmatches;
      results_mtx.unlock();
    }
  }
}

bool comparison(string a, string b) {
  return a < b;
}

vector<string> alphabetize(vector<string> a) {
  sort(a.begin(), a.end(), comparison);
  return a;
}

int main(int argc, char** argv) {
  string keyfile;
  string textfile;
  string outfile;
  int numthreads;
  try {
    keyfile = argv[1];
    textfile = argv[2];
    outfile = argv[3];
    numthreads = stoi(argv[4]);
  } catch(exception& e) {
    cout << "Invalid arguments" << endl;
    return 1;
  }

  // read files
  vector<string> keywords;
  string line = "";
  try {
    ifstream instream(keyfile);
    if(!instream.good()) {
      throw invalid_argument("File does not exist");
    }
    if(instream.is_open()) {
      while(getline(instream, line)) {
        line.pop_back();
        keywords.push_back(line);
      }
    }
    instream.close();
  } catch(exception& e) {
    cout << "Invalid input" << endl;
  }

  vector<string> text;
  line = "";
  try {
    ifstream instream(textfile);
    if(!instream.good()) {
      throw invalid_argument("File does not exist");
    }
    if(instream.is_open()) {
      while(getline(instream,line)) {
        text.push_back(line);
      }
    }
    instream.close();
  } catch(exception& e) {
    cout << "Invalid input" << endl;
  }

  keywords = alphabetize(keywords);

  map<string, int> output;
  for(int i = 0; i < keywords.size(); ++i) {
    output[keywords[i]] = 0;
  }

  vector<thread*> threads;
  for(int i = 0; i < numthreads; ++i) {
    vector<string> linestodo;
    for(int j = 0; j < text.size(); ++j) {
      if(j % numthreads == i) { // when true, thread i will do line j
        linestodo.push_back(text[j]);
      }
    }
    threads.push_back(new thread([&,keywords,linestodo](){
      TextSearch(keywords, linestodo, output);
    }));
  }

  for(int i = 0; i < numthreads; ++i) {
    thread& t = *threads[i];
    t.join();
    delete threads[i];
  }

  threads.resize(0);

  ofstream outstream(outfile);
  map<string, int>::iterator it;
  for(it = output.begin(); it != output.end(); ++it) {
    outstream << it->first << " ";
    outstream << it->second << endl;
  }
}
