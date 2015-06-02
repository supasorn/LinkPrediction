#include <sys/time.h>
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>          // std::mutex
#include <vector>
#include "ceres/ceres.h"
#include "gflags/gflags.h"
using namespace std;

using namespace Eigen;
std::mutex mtx;           // mutex for critical section

DEFINE_double(eta, 0.01, "Mask scale");
DEFINE_double(lambda, 0.01, "Mask scale");
DEFINE_double(lambdaw, 0.01, "Mask scale");
DEFINE_string(movie, "data/movies.mtx", "a");
DEFINE_string(data, "data/ratings_debug_train.mtx", "a");
DEFINE_int32(rank, 5, "a");
DEFINE_int32(lim, 1000, "a");
DEFINE_int32(maxit, 20, "a");
DEFINE_bool(unified, false, "use opengl");

std::map<std::string, double> ticmap;
std::map<std::string, double> ticamap; // accumulate
double timestamp() {
  timeval start; 
  gettimeofday(&start, NULL);
  return ((start.tv_sec) + start.tv_usec/1000000.0);
}

void tic(std::string name) {
  ticmap[name] = timestamp();
}

double toc(std::string name) {
  if (ticmap.find(name) != ticmap.end()) {
    double ret = timestamp() - ticmap[name];
    printf("Time %s: %f\n", name.c_str(), ret);
    return ret;
  }
  return -1;
}

void toca(std::string name) {
  if (ticmap.find(name) != ticmap.end()) {
    if (ticamap.find(name) == ticamap.end())
      ticamap[name] = timestamp() - ticmap[name];
    else
      ticamap[name] += timestamp() - ticmap[name];
  }
}

void tocaList() {
  printf("Profiling\n");
  for (std::map<std::string, double>::iterator i = ticamap.begin(); i != ticamap.end(); i++) {
    printf("  %s : %f\n", i->first.c_str(), i->second);
  }
}

void print_block (int n, char c) {
  // critical section (exclusive access to std::cout signaled by locking mtx):
  mtx.lock();
  for (int i=0; i<n; ++i) { std::cout << c; }
  std::cout << '\n';
  mtx.unlock();
}

struct Movie {
  int id;
  float v;
  Movie(int id, float v) : id(id), v(v) {}
  Movie() {}
};
vector<vector<Movie> > movies;
MatrixXf movieMat;

void loadMovie() {
  FILE *fi = fopen(FLAGS_movie.c_str(), "r");
  char st[200];
  int count = 0;
  while (fgets(st, 200, fi) != NULL) {
    if (st[0] == '%') continue;
    int a, b;
    float c;
    sscanf(st, " %d %d %f", &a, &b, &c);
    if (movies.size() == 0) {
      movies.resize(a);
      movieMat = Matrix<float, Dynamic, Dynamic, RowMajor>(a, 59);
    } else {
      movies[a-1].push_back(Movie(b-1, c));
      movieMat(a-1, b-1) = c;
    }
    printf("%d\n", ++count);
    if (FLAGS_lim > 0 && count > FLAGS_lim) break;
  }
  fclose(fi);
}

struct SparseMatrix {
  int u, m;
  float v;
  SparseMatrix(int u, int m, float v) : u(u), m(m), v(v) {}
  SparseMatrix() {}
};
vector<SparseMatrix> rawRatings;
int nUser, nMovie, nRating;
float avgRating;

MatrixXf L, R, wu, wm;
VectorXf bu, bm;

void loadRating() {
  FILE *fi = fopen(FLAGS_data.c_str(), "r");
  char st[200];
  int count = 0;
  while (fgets(st, 200, fi) != NULL) {
    if (st[0] == '%') continue;
    int a, b;
    float c;
    sscanf(st, " %d %d %f", &a, &b, &c);
    if (nUser == 0) {
      nUser = a;
      nMovie = b;
      nRating = c;
      printf("%d %d %d\n", nUser, nMovie, nRating);
    } else {
      rawRatings.push_back(SparseMatrix(a-1, b-1, c));
      avgRating += c;
      printf("%d\n", ++count);
    }
    if (FLAGS_lim > 0 && count > FLAGS_lim) break;
  }
  avgRating /= count;
  fclose(fi);
}

float RMSE() {
  float se = 0;
  int count = 0;
#pragma omp parallel for
  for (int i = 0; i < rawRatings.size(); i++) {
    SparseMatrix rating = rawRatings[i];
    float e;
    if (FLAGS_unified) {
      e = L.row(rating.u).dot(R.row(rating.m)) + bu(rating.u) + bm(rating.m) + (wu.row(rating.u) + wm.row(rating.m)).dot(movieMat.row(rating.m));
    } else {
      e = L.row(rating.u).dot(R.row(rating.m));
    }
    se += (e - rating.v) * (e - rating.v);
    count ++;
  }
  return sqrt(se / count); 
}

void run() {
  L = Matrix<float,Dynamic,Dynamic,RowMajor>(nUser, FLAGS_rank);
  R = Matrix<float,Dynamic,Dynamic,RowMajor>(nMovie, FLAGS_rank);
  wu = Matrix<float,Dynamic,Dynamic,RowMajor>::Zero(nUser, 59);
  wm = Matrix<float,Dynamic,Dynamic,RowMajor>::Zero(nMovie, 59);
  bu = VectorXf::Zero(nUser);
  bm = VectorXf::Zero(nMovie);

  tic("a");
  for (int i = 0; i < nUser; i++) 
    for (int j = 0; j < FLAGS_rank; j++) 
      L(i, j) = ((double) rand() / (RAND_MAX)) * sqrt(avgRating / FLAGS_rank / 0.25);
  for (int i = 0; i < nMovie; i++) 
    for (int j = 0; j < FLAGS_rank; j++) 
      R(i, j) = ((double) rand() / (RAND_MAX)) * sqrt(avgRating / FLAGS_rank / 0.25);
  toc("a");
  tic("Total");
  for (int it = 0; it < FLAGS_maxit; it++) {
    printf("Iteration %d\n", it);
    tic("one iteration");
    for (auto rating : rawRatings) {
      if (FLAGS_unified) {
        float c1 = (1 - FLAGS_eta * FLAGS_lambda);
        float c2 = (1 - FLAGS_eta * FLAGS_lambdaw);
        float e = L.row(rating.u).dot(R.row(rating.m)) 
          + bu(rating.u) + bm(rating.m)
          + (wu.row(rating.u) + wm.row(rating.m)).dot(movieMat.row(rating.m));
        e -= rating.v;

        auto LT = L.row(rating.u);
        L.row(rating.u) = c1 * L.row(rating.u) - FLAGS_eta * e * R.row(rating.m);
        R.row(rating.m) = c1 * R.row(rating.m) - FLAGS_eta * e * LT;

        bu(rating.u) -= FLAGS_eta * e;
        bm(rating.m) -= FLAGS_eta * e;

        wu.row(rating.u) = c2 * wu.row(rating.u) - FLAGS_eta * e * movieMat.row(rating.m);
        wm.row(rating.m) = c2 * wm.row(rating.m) - FLAGS_eta * e * movieMat.row(rating.m);
      } else {
        float c1 = (1 - FLAGS_eta * FLAGS_lambda);
        float e = L.row(rating.u).dot(R.row(rating.m));
        e -= rating.v;
        auto LT = L.row(rating.u);
        L.row(rating.u) = c1 * L.row(rating.u) - FLAGS_eta * e * R.row(rating.m);
        R.row(rating.m) = c1 * R.row(rating.m) - FLAGS_eta * e * LT;
      }
    }
    toc("one iteration");
    printf("RMSE %f\n", RMSE());
  }
  toc("Total");
  //L(0, 0) = 1;
  //L(0, 1) = 2;
  //L(0, 2) = 3;
  //L(0, 3) = 4;
  //L(0, 4) = 5;
  //L.row(2) = L.row(1);
  //cout << L.row(2) << endl;
}


void bench() {
  Matrix<float,Dynamic,Dynamic,RowMajor> A(10000, 10000);
  tic("a");
  for (int i = 0; i < 1000; i++) {
    int a = rand()%1000;
    int b = rand()%1000;
    A.row(a) = A.row(a) * A.row(a).dot(A.row(b));
  }
  toc("a");
  tic("b");
  for (int i = 0; i < 1000; i++) {
    int a = rand()%1000;
    int b = rand()%1000;
    A.col(a) = A.col(a) * A.col(a).dot(A.col(b));
  }
  toc("a");
  exit(0);
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  //bench();
  loadMovie();
  loadRating();

  run();
  //std::thread th1 (print_block,5000,'*');
  //std::thread th2 (print_block,5000,'$');

  //th1.join();
  //th2.join();

  return 0;
}
