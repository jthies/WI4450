CXX=g++
CXX_FLAGS=-O2 -g -fopenmp

DEFS=-DNDEBUG

#default target (built when typing just "make")
default: run_tests.x cg_poisson.x

# general rule to comple a C++ source file into an object file
%.o: %.cpp
	${CXX} -c ${CXX_FLAGS} ${DEFS} $<

#define some dependencies on headers
operations.o: operations.hpp
cg_solver.o: cg_solver.hpp operations.hpp
cg_poisson.o: cg_solver.hpp operations.hpp
gtest_mpi.o: gtest_mpi.hpp

TEST_SOURCES=test_operations.cpp
MAIN_OBJ=cg_poisson.o cg_solver.o operations.o

run_tests.x: run_tests.cpp ${TEST_SOURCES} gtest_mpi.o operations.o
	${CXX} ${CXX_FLAGS} ${DEFS} -o run_tests.x $^

cg_poisson.x: ${MAIN_OBJ}
	${CXX} ${CXX_FLAGS} ${DEFS} -o cg_poisson.x $^


PHONY: clean

clean:
	-rm *.o *.x
