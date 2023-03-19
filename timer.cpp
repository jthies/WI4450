
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
    flops_[label_] = m; //m is double
    bytes_[label_] = b*sizeof(double);  //b is double, sizeof(double) is integer, double*integer=double
                                        //all basis operations use doubles in computations
  }

void Timer::summarize(std::ostream& os)
{
  
  os << "==================== TIMER SUMMARY =========================================" << std::endl;
  os << "label               \tcalls     \ttotal time\tmean time\tcomp. intensity\t mean Gflop/s\tmean Gbyte/s"<<std::endl;
  os << "----------------------------------------------" << std::endl;
  for (auto [label, time]: times_)
  {
    int count = counts_[label];
    double gflops = flops_[label] / 1000000.0; //convert flop to Gflop
    double gbytes = bytes_[label] / 1000000.0; //convert byte to Gbyte
    double mean_time = time/double(count);
    std::cout << std::setw(20) << label << "\t" << std::setw(10) << count << "\t" << std::setw(10) << time << "\t" << std::setw(10) 
        << mean_time << "\t" << std::setw(10) << gflops/gbytes << "\t" << std::setw(10) << gflops/mean_time <<"\t" << std::setw(10) 
        << gbytes/mean_time << std::endl;
  }
  os << "============================================================================" << std::endl;
}
