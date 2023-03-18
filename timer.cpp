
#include "timer.hpp"

#include <iomanip>

// for timing routine
#include <omp.h>

#include <iostream>
#include <iomanip>

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
    flops_[label_] = n;
    bytes_[label_] = b*n*size(double);
  }

void Timer::summarize(std::ostream& os)
{
  
  os << "==================== TIMER SUMMARY =========================================" << std::endl;
  os << "label               \tcalls     \ttotal time\tmean time \tflops"<<std::endl;
  os << "----------------------------------------------" << std::endl;
  for (auto [label, time]: times_)
  {
    int count = counts_[label];
    double flops = flops_[label];
    double bytes = bytes_[label];
    std::cout << std::setw(20) << label << "\t" << std::setw(10) << count << "\t" << std::setw(10) << time << "\t" << std::setw(10) << time/double(count) << "\t" << std::setw(10) << flops << "\t" << std::setw(10) << bytes << std::endl;
  }
  os << "============================================================================" << std::endl;
}
