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

DEFINE_double(eta, 0.01, "");
DEFINE_double(lambda, 0.01, "");
DEFINE_double(lambdaw, 0.01, "");
DEFINE_bool(big, false, "");
DEFINE_bool(cold, false, "");
DEFINE_int32(rank, 5, "");
DEFINE_int32(lim, 0, "");
DEFINE_int32(maxit, 20, "");
DEFINE_int32(cores, 0, "");
DEFINE_int32(interval, 5, "");
DEFINE_string(method, "SGD", "");
DEFINE_bool(unified, true, "");
DEFINE_bool(shuffle, true, "");
DEFINE_bool(byit, true, "");
DEFINE_bool(onermse, false, "");

DEFINE_string(movie, "", "a");
DEFINE_string(data, "", "a");
DEFINE_string(datatest, "", "a");

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
int nUser, nMovie, nRating, j;
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
      movieMat = Matrix<float, Dynamic, Dynamic, RowMajor>::Zero(a, b);
    } else {
      movieMat(shuffleR[a-1], b-1) = c;
      if (fabs(c) > 1) {
        printf("err\n");
        exit(0);
      }
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
      if (c > 5 || c < 0) printf("WRONG INPUT RATING!!");
      //c = (c-0.5)/4.5;

      rawRatings.push_back(SparseMatrix(shuffleL[a-1], shuffleR[b-1], c));
      avgRating += c;
      count++;
      //mm[b-1]++;
    }
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
      //double a = L.row(rating.u).dot(R.row(rating.m));
      //printf("%f %f\n", L.row(rating.u).dot(L.row(rating.u)), R.row(rating.m).dot(R.row(rating.m)));
      //if (a != a) {
        //cout << L.row(rating.u) << endl;
        //cout << R.row(rating.m) << endl;
        //printf("err\n");
        //exit(0);
      //}
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

float rn(MatrixXf &m, int r) {
  return sqrt(m.row(r).dot(m.row(r)));
}

float vn(VectorXf &v) {
  return sqrt(v.dot(v));
}

void update(SparseMatrix &rating) {
  if (FLAGS_unified) {
    float c1 = (1 - FLAGS_eta * FLAGS_lambda);
    float c2 = (1 - FLAGS_eta * FLAGS_lambdaw);
    float LuRm = L.row(rating.u).dot(R.row(rating.m));
    float e = LuRm + bu(rating.u) + bm(rating.m)
      + (wu.row(rating.u) + wm.row(rating.m)).dot(movieMat.row(rating.m));
    e -= rating.v;

    /*
    if (fabs(rn(movieMat, rating.m) - 1) > 0.00001 ) {
      printf("err %f\n", rn(movieMat, rating.m) - 1);
      for (int i = 0; i < 59; i++) {
        printf("%f\n", movieMat(rating.m, i));
      }
      printf("rating.m = %d\n", rating.m);
      exit(0);
    }

    if (fabs(e) > 1.5 || e != e) {
      printf("#%d [u=%6d, m=%6d]   ", j, rating.u, rating.m);

      printf("e=%4.2f, LuRm=%4.2f |Lu|=%4.2f, |Rm|=%4.2f, |wu|=%4.2f, |wm|=%4.2f, |f(m)|=%.4f, |bu|=%4.2f, |bm|=%4.2f\n", 
        e, LuRm, rn(L,rating.u), rn(R,rating.m),  rn(wu, rating.u), rn(wm, rating.m), rn(movieMat, rating.m), bu(rating.u), bm(rating.m));

      // cout << "Lu" << L.row(rating.u) << endl;
      // cout << "Rm" << R.row(rating.m) << endl;
      // cout << "wu" << wu.row(rating.u) << endl;
      // cout << "wm" << wm.row(rating.m) << endl;
      if (fabs(e) > 100) exit(-1);
    }*/

    MatrixXf LT(L.row(rating.u));
    L.row(rating.u) = c1 * L.row(rating.u) - FLAGS_eta * e * R.row(rating.m);
    R.row(rating.m) = c1 * R.row(rating.m) - FLAGS_eta * e * LT;
    float t;
    /*
    if ((t = L.row(rating.u).dot(L.row(rating.u))) > 900) {
#pragma omp critical
      printf("t = %f\n", t);
      for (int i = 0; i < FLAGS_rank; i++) {
        printf("%f ", L(rating.u, i));
      }
      printf("\n");
      exit(0);
    }
    if ((t = R.row(rating.m).dot(R.row(rating.m))) > 900) {
#pragma omp critical
      printf("t = %f\n", t);
      for (int i = 0; i < FLAGS_rank; i++) {
        printf("%f ", R(rating.m, i));
      }
      printf("\n");
      exit(0);
    }*/

    bu(rating.u) -= FLAGS_eta * e;
    bm(rating.m) -= FLAGS_eta * e;

    wu.row(rating.u) = c2 * wu.row(rating.u) - FLAGS_eta * e * movieMat.row(rating.m);
    wm.row(rating.m) = c2 * wm.row(rating.m) - FLAGS_eta * e * movieMat.row(rating.m);

    float nLu = rn(L,rating.u);
    float nRm = rn(R,rating.m);
    float nwu = rn(wu, rating.u);
    float nwm = rn(wm, rating.m);

    //if ( fabs(bu(rating.u)) > 1.0 || fabs(bm(rating.m)) > 1.0 || nLu > 1.4 || nRm > 1.4 || nwu>1 || nwm > 1) {
      //float LuRm = L.row(rating.u).dot(R.row(rating.m));
      //printf("#%d [u=%6d, m=%6d]   ", j, rating.u, rating.m);
      //printf("e=%.2f, LuRm=%4.2f, |Lu|=%.3f, |Rm|=%.3f, |wu|=%.3f, |wm|=%.3f, |bu|=%.3f, |bm|=%.3f\n", 
        //e, LuRm, nLu, nRm, nwu, nwm, bu(rating.u), bm(rating.m));
    //}

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
  printf(" i,   RMSE, time\n");
  printf(" 0, %.4f,    0  \n", RMSE(rawRatings));
  for (int it = 0; it < FLAGS_maxit; it++) {
    printf("Iteration %d\n", it);
    tic("one iteration");
    j=0;
    for (auto rating : rawRatings) {
      
      update(rating);
      j++;
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
  printf(" i,   RMSE, time\n");
  printf(" 0, %.4f,    0  \n", RMSE(rawRatings));
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
    if (!FLAGS_onermse) {
      if (FLAGS_byit) {
        //printf("[%d] RMSE %f:%f [%fs, %fs]\n", it+1, RMSE(rawRatings), RMSE(rawRatingsTest), timestamp() - start1, timestamp() - start0);
        printf("%2d, %.4f, %f \n", it+1, RMSE(rawRatings), timestamp() - start0);
      } else {
        if (timestamp() - start0 > interval) {
          interval += FLAGS_interval;
          printf("%2d, %.4f, %f \n", it+1, RMSE(rawRatings), timestamp() - start0);
        }
      }
    } else {
      printf("%2d, %f \n", it+1, timestamp() - start0);
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
  printf(" i,   RMSE, time\n");
  printf(" 0, %.4f,    0  \n", RMSE(rawRatings));

  vector<thread> threads;
  for (int i = 0; i < FLAGS_cores; i++) {
    threads.push_back(thread(NOMADThread, i));
  }
  double start0 = timestamp();
  int interval = FLAGS_interval;
  for (int i = 0; i < FLAGS_maxit; i++) {
    if (!FLAGS_onermse) {
      if (timestamp() - start0 > interval) {
        interval += FLAGS_interval;
        printf("%2d, %.4f, %f \n", i+1, RMSE(rawRatings), timestamp() - start0);
      }
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

void test() {
  for (int i = 0; i < 59; i++) {
    printf("%f\n", movieMat(92, i));
  }
  printf("rating.m = %d\n", 92);
}


int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_cores == 0) {
    FLAGS_cores = std::thread::hardware_concurrency();
  }

  if (FLAGS_cold) {
    FLAGS_movie = "data/movies.mtx";
    FLAGS_data = "data/ratings_cs_train.mtx";
    FLAGS_datatest = "data/ratings_cs_test.mtx";
  } else if (FLAGS_big) {
    FLAGS_movie = "data/movies.mtx";
    FLAGS_data = "data/ratings_train.mtx";
    FLAGS_datatest = "data/ratings_test.mtx";
  } else {
    FLAGS_movie = "data/movies_ratings_debug.mtx";
    FLAGS_data = "data/ratings_debug_train.mtx";
    FLAGS_datatest = "data/ratings_debug_test.mtx";
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
