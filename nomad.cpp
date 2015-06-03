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

DEFINE_double(eta, 0.01, "Mask scale");
DEFINE_double(lambda, 0.01, "Mask scale");
DEFINE_double(lambdaw, 0.01, "Mask scale");
DEFINE_string(movie, "data/movies.mtx", "a");
DEFINE_string(data, "data/ratings_debug_train.mtx", "a");
DEFINE_string(datatest, "data/ratings_debug_test.mtx", "a");
DEFINE_int32(rank, 5, "a");
DEFINE_int32(lim, 0, "a");
DEFINE_int32(maxit, 20, "a");
DEFINE_int32(cores, 0, "a");
DEFINE_int32(interval, 5, "a");
DEFINE_string(method, "SGD", "a");
DEFINE_bool(unified, true, "use opengl");
DEFINE_bool(shuffle, true, "use opengl");
DEFINE_bool(byit, false, "use opengl");

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


//struct Movie {
  //int id;
  //float v;
  //Movie(int id, float v) : id(id), v(v) {}
  //Movie() {}
//};
MatrixXf movieMat;
vector<int> shuffleL, shuffleR;

struct SparseMatrix {
  int u, m;
  float v;
  SparseMatrix(int u, int m, float v) : u(u), m(m), v(v) {}
  SparseMatrix() {}
};
vector<SparseMatrix> rawRatings;
vector<SparseMatrix> rawRatingsTest;
int nUser, nMovie, nRating;
float avgRating;

MatrixXf L, R, wu, wm;
VectorXf bu, bm;

void printParameters() {
  printf(" Unified: %d\n", FLAGS_unified);
  printf(" Rank: %d\n", FLAGS_rank);
  printf(" Lambda LR: %f\n", FLAGS_lambda);
  printf(" Lambda W: %f\n", FLAGS_lambdaw);
  printf(" Eta: %f\n", FLAGS_eta);
  printf(" Cores: %d\n", FLAGS_cores);
  printf(" Method: %s\n", FLAGS_method.c_str());
  printf(" Train: %s\n", FLAGS_data.c_str());
  printf(" Test: %s\n", FLAGS_datatest.c_str());
  printf(" Shuffle: %d\n---------------\n", FLAGS_shuffle);
}

void loadMovie() {
  FILE *fi = fopen(FLAGS_movie.c_str(), "r");
  char st[200];
  int count = 0;
  while (fgets(st, 200, fi) != NULL) {
    if (st[0] == '%') continue;
    int a, b;
    float c;
    sscanf(st, " %d %d %f", &a, &b, &c);
    if (count == 0) {
      movieMat = Matrix<float, Dynamic, Dynamic, RowMajor>(a, b);
    } else {
      movieMat(shuffleR[a-1], b-1) = c;
    }
    count++;
    if (FLAGS_lim > 0 && count > FLAGS_lim) break;
  }
  fclose(fi);
  printf("Reading Movie Tags .. Done\n");
}

void shuffle() {
  shuffleL.resize(nUser);
  for (int i = 0; i < nUser; i++)
    shuffleL[i] = i;
  shuffleR.resize(nMovie);
  for (int i = 0; i < nMovie; i++)
    shuffleR[i] = i;
  if (FLAGS_shuffle) {
    random_shuffle(shuffleR.begin(), shuffleR.end());
    random_shuffle(shuffleL.begin(), shuffleL.end());
  }
}

void load() {
  FILE *fi = fopen(FLAGS_data.c_str(), "r");
  char st[200];
  int count = 0;
  //map<int, int> mm;
  while (fgets(st, 200, fi) != NULL) {
    if (st[0] == '%') continue;
    int a, b;
    float c;
    sscanf(st, " %d %d %f", &a, &b, &c);
    if (nUser == 0) {
      nUser = a;
      nMovie = b;
      nRating = c;
      shuffle();
      printf("User: %d Movies: %d Ratings: %d\n", nUser, nMovie, nRating);
    } else {
      rawRatings.push_back(SparseMatrix(shuffleL[a-1], shuffleR[b-1], c));
      avgRating += c;
      count++;
      //mm[b-1]++;
    }
    if (FLAGS_lim > 0 && count > FLAGS_lim) break;
  }
  avgRating /= count;
  printf("Reading Training Set .. Done\n");
  fclose(fi);
  fi = fopen(FLAGS_datatest.c_str(), "r");
  count = 0;
  while (fgets(st, 200, fi) != NULL) {
    if (st[0] == '%') continue;
    int a, b;
    float c;
    sscanf(st, " %d %d %f", &a, &b, &c);
    if (count > 0) {
      rawRatingsTest.push_back(SparseMatrix(shuffleL[a-1], shuffleR[b-1], c));
    }
    count++;
    if (FLAGS_lim > 0 && count > FLAGS_lim) break;
  }
  fclose(fi);
  printf("Reading Test Set .. Done\n");

  loadMovie();
  //for(auto &it1 : mm) {
    //printf("%d %d\n", it1.first, it1.second);
  //}
}

float RMSE(vector<SparseMatrix> &ratings) {
  //tic("RMSE");
  double se = 0;
  int count = 0;
#pragma omp parallel for
  for (int i = 0; i < ratings.size(); i++) {
    SparseMatrix rating = ratings[i];
    float e;
    if (FLAGS_unified) {
      e = L.row(rating.u).dot(R.row(rating.m)) + bu(rating.u) + bm(rating.m) + (wu.row(rating.u) + wm.row(rating.m)).dot(movieMat.row(rating.m));
    } else {
      e = L.row(rating.u).dot(R.row(rating.m));
    }                                             
#pragma omp critical 
    {
      se += double(e - rating.v) * double(e - rating.v);
      count ++;
    }
  }
  //toc("RMSE");
  return sqrt(se / count); 
}

void update(SparseMatrix &rating) {
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

void init() {
  L = Matrix<float,Dynamic,Dynamic,RowMajor>(nUser, FLAGS_rank);
  R = Matrix<float,Dynamic,Dynamic,RowMajor>(nMovie, FLAGS_rank);
  wu = Matrix<float,Dynamic,Dynamic,RowMajor>::Zero(nUser, 59);
  wm = Matrix<float,Dynamic,Dynamic,RowMajor>::Zero(nMovie, 59);
  bu = VectorXf::Zero(nUser);
  bm = VectorXf::Zero(nMovie);

  printf("Average Rating = %f\n", avgRating);
  tic("a");
  for (int i = 0; i < nUser; i++) 
    for (int j = 0; j < FLAGS_rank; j++) 
      L(i, j) = ((double) rand() / (RAND_MAX)) * sqrt(avgRating / FLAGS_rank / 0.25);
  for (int i = 0; i < nMovie; i++) 
    for (int j = 0; j < FLAGS_rank; j++) 
      R(i, j) = ((double) rand() / (RAND_MAX)) * sqrt(avgRating / FLAGS_rank / 0.25);

}
void run() {
  printf("RMSE %f\n", RMSE(rawRatings));
  for (int it = 0; it < FLAGS_maxit; it++) {
    printf("Iteration %d\n", it);
    tic("one iteration");
    for (auto rating : rawRatings) {
      update(rating);
    }
    toc("one iteration");
    printf("RMSE %f\n", RMSE(rawRatings));
  }
}


void DSGD() {
  vector<vector<vector<SparseMatrix> > > subRawRatings(FLAGS_cores);
  for (int i = 0; i < subRawRatings.size(); i++) {
    subRawRatings[i].resize(FLAGS_cores);
  }
  float divu = 1.0 * nUser / FLAGS_cores;
  float divm = 1.0 * nMovie / FLAGS_cores;

  for (auto rating : rawRatings) {
    subRawRatings[rating.u / divu][rating.m / divm].push_back(rating);
  }

  //for (int i = 0; i < FLAGS_cores; i++, printf("\n")) {
    //for (int j = 0; j < FLAGS_cores; j++) {
      //printf("%d ", subRawRatings[i][j].size());
    //}
  //} 

  printf("[0] RMSE %f\n", RMSE(rawRatings));
  int interval = FLAGS_interval;
  double start0 = timestamp();
  for (int it = 0; it < FLAGS_maxit; it++) {
    //double start1 = timestamp();
    for (int j = 0; j < FLAGS_cores; j++) {
#pragma omp parallel for
      for (int k = 0; k < FLAGS_cores; k++) {
        for (auto rating: subRawRatings[k][(j + k) % FLAGS_cores]) {
          update(rating);
        }
      } 
    }
    if (FLAGS_byit) {
      //printf("[%d] RMSE %f:%f [%fs, %fs]\n", it+1, RMSE(rawRatings), RMSE(rawRatingsTest), timestamp() - start1, timestamp() - start0);
      printf("[%d] RMSE %f [%fs]\n", it+1, RMSE(rawRatings), timestamp() - start0);
    } else {
      if (timestamp() - start0 > interval) {
        interval += FLAGS_interval;
        printf("[%d] RMSE %f [%fs]\n", it+1, RMSE(rawRatings), timestamp() - start0);
      }
    }
  }
}

vector<queue<int> > qs;
mutex mtx[100];

vector<vector<vector<SparseMatrix> > > subRawRatings;
int halt;
void NOMADThread(int id) {
  while (!halt) {
    while (!qs[id].empty() && !halt) {
      mtx[id].lock();
      int s = qs[id].front();
      qs[id].pop();
      mtx[id].unlock();

      //printf("%d %d\n", id, s);
      for (auto rating : subRawRatings[id][s]) {
        update(rating);
      }

      //int next = rand() % FLAGS_cores;
      int next = 0;
      for (int i = 1; i < qs.size(); i++) {
        if (qs[i].size() < qs[next].size())
          next = i;
      }

      mtx[next].lock();
      qs[next].push(s);
      mtx[next].unlock();
    }
  }
}

void NOMAD() {
  subRawRatings.resize(FLAGS_cores);
  int s2 = FLAGS_cores * 10;
  for (int i = 0; i < subRawRatings.size(); i++) {
    subRawRatings[i].resize(s2);
  }

  float divu = 1.0 * nUser / FLAGS_cores;
  float divm = 1.0 * nMovie / s2;

  for (auto rating : rawRatings) {
    subRawRatings[rating.u / divu][rating.m / divm].push_back(rating);
  }

  qs.resize(FLAGS_cores);
  for (int i = 0; i < subRawRatings[0].size(); i++) {
    qs[rand() % FLAGS_cores].push(i);
  }

  printf("[0] RMSE %f\n", RMSE(rawRatings));

  vector<thread> threads;
  for (int i = 0; i < FLAGS_cores; i++) {
    threads.push_back(thread(NOMADThread, i));
  }
  double start0 = timestamp();
  int interval = FLAGS_interval;
  for (int i = 0; i < FLAGS_maxit; i++) {
    if (timestamp() - start0 > interval) {
      interval += FLAGS_interval;
      printf("[%d] RMSE %f [%fs]\n", i+1, RMSE(rawRatings), timestamp() - start0);
    }
    sleep(1);
  }

  halt = 1;
  for (int i = 0; i < FLAGS_cores; i++) {
    threads[i].join();
  }

  //for (int i = 0; i < subRawRatings.size(); i++, printf("\n")) {
    //for (int j = 0; j < subRawRatings[i].size(); j++) {
      //printf("%d ", subRawRatings[i][j].size());
    //}
  //} 
  


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

  if (FLAGS_cores == 0) {
    FLAGS_cores = std::thread::hardware_concurrency();
  }
  srand(0);
  load();
  init();
  printParameters();

  if (FLAGS_method == "NOMAD")
    NOMAD();
  else if (FLAGS_method == "DSGD")
    DSGD();
  else
    run();

  double trainRMSE = RMSE(rawRatings);
  double testRMSE = RMSE(rawRatingsTest);
  printf("Train RMSE :%f\nTest RMSE :%f\n", trainRMSE, testRMSE);

  //FILE *fo = fopen("output/crossvalidation.txt", "a"); 
  //fprintf(fo, "%f %f %f %f %f\n", FLAGS_rank, FLAGS_lambda, FLAGS_lambdaw, trainRMSE, testRMSE);
  //fclose(fo);
  //std::thread th1 (print_block,5000,'*');
  //std::thread th2 (print_block,5000,'$');

  //th1.join();
  //th2.join();

  return 0;
}
