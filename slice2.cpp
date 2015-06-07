#include <sys/time.h>
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>          // std::mutex
#include <vector>
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include <queue>
#include <unistd.h>
using namespace std;

using namespace Eigen;

string data = "data/ml-20m/ratings.csv";

void head(FILE *fo) {
  fprintf(fo, "\%\%\%MatrixMarket matrix coordinate real general\n\%\n138493 27278 -1\n");
}


struct SparseMatrix {
  int u, m;
  float v;
  SparseMatrix(int u, int m, float v) : u(u), m(m), v(v) {}
  SparseMatrix() {}
};
vector<SparseMatrix> r;
int nUser, nMovie, nRating = 0;

void load(string f) {
  FILE *fi = fopen(f.c_str(), "r");
  char st[200];
  int count = 0;
  while (fgets(st, 200, fi) != NULL) {
    if (st[0] == '%') continue;
    int a, b;
    float c;
    sscanf(st, " %d %d %f", &a, &b, &c);
    if (count == 0) {
      nUser = a;
      nMovie = b;
      nRating += c;
      printf("%d %d %f\n", a, b, c);
    } else {
      r.push_back(SparseMatrix(a-1, b-1, c));
    }
    count++;
  }
  fclose(fi);
}

vector<int> rmov, ruser;
void initSlice() {
  rmov.resize(nMovie);
  ruser.resize(nUser);
  for (int i = 0; i < nMovie; i++) 
    rmov[i] = i;
  for (int i = 0; i < nUser; i++) 
    ruser[i] = i;

  random_shuffle(rmov.begin(), rmov.end());
  random_shuffle(ruser.begin(), ruser.end());
}
void slice(int num) {
  set<int> inTest;
  for (int i = 0; i < 0.2 * nMovie; i++) {
    inTest.insert(rmov[i]);
  }
  set<int> outBlock;
  for (int i = 0; i < (100 - num) / 100.0 * nUser; i++) {
    outBlock.insert(ruser[i]);
  }

  FILE *ftrain = fopen(("data/ratings_train_" + to_string(num) + ".mtx").c_str(), "w");
  FILE *ftest = fopen(("data/ratings_test_" + to_string(num) + ".mtx").c_str(), "w");
  head(ftrain);
  head(ftest);
  for (int i = 0; i < r.size(); i++) {
    if (inTest.find(r[i].m) != inTest.end()) {
      if (outBlock.find(r[i].u) != outBlock.end()) {
        fprintf(ftrain, "%d %d %.1f\n", r[i].u + 1, r[i].m + 1, r[i].v);
      } else {
        fprintf(ftest, "%d %d %.1f\n", r[i].u + 1, r[i].m + 1, r[i].v);
      }
    } else {
      fprintf(ftrain, "%d %d %.1f\n", r[i].u + 1, r[i].m + 1, r[i].v);
    }
  }

  fclose(ftrain);
  fclose(ftest);
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  srand(0);
  load("data/ratings_train.mtx");
  printf("%d\n", r.size());
  load("data/ratings_test.mtx");
  printf("%d\n", r.size());
  
  initSlice();
  slice(60);
  slice(70);
  slice(80);
  slice(90);
}
