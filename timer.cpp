
#include "timer.hpp"

#include <iomanip>

// for timing routine
#include <omp.h>

#include <iostream>
#include <iomanip>
#include <cmath>

// static members of a class must be defined
// somewhere in an object file, otherwise you
// will get linker errors (undefined reference)
std::map<std::string, int> Timer::counts_;
std::map<std::string, double> Timer::times_;
std::map<std::string, double> Timer::flops_;
std::map<std::string, double> Timer::bytes_;

  Timer::Timer(std::string label)
  : label_(label)
  {
    t_start_ = omp_get_wtime();
  }


  Timer::~Timer()
  {
    double t_end = omp_get_wtime();
    times_[label_] += t_end - t_start_;
    counts_[label_]++;
    flops_[label_] = m; //m is double
    bytes_[label_] = b;  //b is double
  }

void Timer::summarize(std::ostream& os)
{
  
  os << "==================== TIMER SUMMARY =========================================" << std::endl;
  os << "label               \tcalls     \ttotal time\tmean time\tcomp. intensity\t mean Gflop/s\tmean Gbyte/s"<<std::endl;
  os << "----------------------------------------------" << std::endl;
  double convert = std::pow(10.0,9);
  unsigned int n_threads = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 1);
  for (auto [label, time]: times_)
  {
    int count = counts_[label];
    int itPerThread = int(count) / int(n_threads);
    double gflops = flops_[label] / convert; //convert flop to Gflop
    double gbytes = bytes_[label] / convert; //convert byte to Gbyte
    double gflops_rate = gflops * itPerThread / time;
    double gbytes_rate = gbytes * itPerThread / time;
    double mean_time = time/double(count);
    std::cout << std::setw(20) << label << "\t" << std::setw(10) << count << "\t" << std::setw(10) << time << "\t" << std::setw(10) 
        << mean_time << "\t" << std::setw(10) << gflops/gbytes << "\t" << std::setw(10) << gflops_rate <<"\t" << std::setw(10) 
        << gbytes_rate << std::endl;
  }
  os << "============================================================================" << std::endl;
}
