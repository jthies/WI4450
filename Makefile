CXX=g++
CXX_FLAGS=-O2 -g -fopenmp

DEFS=

# general rule to comple a C++ source file into an object file
%.o: %.cpp
	${CXX} -c ${CXX_FLAGS} $<

# general rule to comple a C++ source file into an executable (.x used here)
%.x: %.o
	${CXX} ${CXX_FLAGS} -o $@ $<


#define some dependencies on headers
operations.o: operations.hpp
cg_solver.o: cg_solver.hpp operations.hpp
gtest_mpi.o: gtest_mpi.hpp

TEST_SOURCES=test_operations.cpp

run_tests.x: run_tests.cpp ${TEST_SOURCES} gtest_mpi.o operations.o
	${CXX} ${CXX_FLAGS} -o run_tests.x $^


PHONY: clean

clean:
	-rm *.o *.x
