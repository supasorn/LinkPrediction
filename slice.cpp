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
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

void head(FILE *fo) {
  fprintf(fo, "\%\%\%MatrixMarket matrix coordinate real general\n\%\n138493 27278 -1\n");
}
void load() {
  FILE *fi = fopen(data.c_str(), "r");
  char st[300];
  int count = 0;
  //map<int, int> mm;
  FILE *fo1 = fopen("data/train_mf.mtx", "w");
  FILE *fo2 = fopen("data/test_mf.mtx", "w");
  FILE *fo3 = fopen("data/train_cs.mtx", "w");
  FILE *fo4 = fopen("data/test_cs.mtx", "w");
  head(fo1);
  head(fo2);
  head(fo3);
  head(fo4);

  vector<int> cols(27278);
  for (int i = 0; i < 27278; i++) {
    cols[i] = rand() % 10 < 8;
  }

  map<int, int> userMap, movieMap;

  int userCount = 0, movieCount = 0;
  while (fgets(st, 300, fi) != NULL) {
    count++;
    string sst = st;
    if (count == 1) continue;
    //printf("%s\b", sst.c_str());
    vector<string> sp;
    split(sst, ',', sp);
    int user = stoi(sp[0]);
    int movie = stoi(sp[1]);
    float rat = stof(sp[2]);

    if (userMap.find(user) == userMap.end()) {
      userMap[user] = ++userCount;
    }

    if (movieMap.find(movie) == movieMap.end()) {
      movieMap[movie] = ++movieCount;
    }

    user = userMap[user];
    movie = movieMap[movie];
    //printf("%d %d %f\n", stoi(sp[0]), stoi(sp[1]), stof(sp[2]));

    //printf("%d\n", movie);

    if (rand() % 10 < 8)
      fprintf(fo1, "%d %d %f\n", user, movie, rat);
    else
      fprintf(fo2, "%d %d %f\n", user, movie, rat);
  
    if (cols[movie - 1])
      fprintf(fo3, "%d %d %f\n", user, movie, rat);
    else
      fprintf(fo4, "%d %d %f\n", user, movie, rat);

    if (count % 1000 == 0) printf("%d\n", count);
    //if (count > 100) break;
  }

  fclose(fo1);
  fclose(fo2);
  fclose(fo3);
  fclose(fo4);
}
int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  srand(0);
  load();
}
