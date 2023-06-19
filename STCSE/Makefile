CXX=g++
CXX_FLAGS=-O2 -g -fopenmp -std=c++17

DEFS=-DNDEBUG

#default target (built when typing just "make")
default: main.x 

# general rule to comple a C++ source file into an object file
%.o: %.cpp
	${CXX} -c ${CXX_FLAGS} ${DEFS} $<

#define some dependencies on headers
operations.o: operations.hpp
cg_solver.0: cg_solver.hpp operations.hpp
fe.o: fe.hpp operations.hpp 
be.o: be.hpp operations.hpp cg_solver.hpp
gmres.o: gmres.hpp operations.hpp timer.hpp
preconditioned_gmres.o: preconditioned_gmres.hpp operations.hpp timer.hpp

MAIN=main.o fe.o be.o gmres.o preconditioned_gmres.o operations.o timer.o cg_solver.o

main.x: ${MAIN} 
	${CXX} ${CXX_FLAGS} ${DEFS} -o main.x $^

clean:
	-rm *.o *.x

# phony targets are run regardless of dependencies being up-to-date
PHONY: clean, test
