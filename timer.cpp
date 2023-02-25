
#include "timer.hpp"

#include <iomanip>

// for timing routine
#include <omp.h>

// static members of a class must be defined
// somewhere in an object file, otherwise you
// will get linker errors (undefined reference)
std::map<std::string, int> Timer::counts_;
std::map<std::string, double> Timer::times_;

  Timer::Timer(std::string label)
  : label_(label)
  {
    t_start_ = omp_get_wtime();
  }


  Timer::~Timer()
  {
    double t_end = omp_get_wtime();
    times_[label_]  += (t_end - t_start_);
    counts_[label_]++;
  }

void Timer::summarize(std::ostream& os)
{
  os << "===== TIMER SUMMARY ===============================" << std::endl;
  os << " label    \t calls \ttotal time\t mean time"<<std::endl;
  os << "---------------------------------------------------" << std::endl;
  for (auto [label, time]: times_)
  {
    int count = counts_[label];
    std::cout << std::setw(10) << label << "\t" << std::setw(7) << count << "\t" << std::setw(10) << time << "\t" << std::setw(10) << time/double(count) << std::endl;
  }
  os << "===================================================" << std::endl;
}
