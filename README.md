# Special Topics in CSE: High Performance Computing for Linear Algebra

## Homework 1: Conjugate Gradient Method

In this exercise, we will exploit **shared memory parallelism** to solve
a standard partial differential equation, the 3D Poisson equation:

```latex
\frac{\partial^2 u}{\partial x^2} + 
\frac{\partial^ u2}{\partial y^2} + 
\frac{\partial^2 u}{\partial z^2} + 
= f(x,y,z),
```
on a unit cube, subject to Dirichlet boundary conditions


1. implement the missing functions ``axpby``, ``dot``,and ``matvec``  parallelized for shared memory using OpenMP.
The specification of these functions can be found in the file ``src/operations.c``.
Implement suitable unit tests to verify they work as expected. Remember that a good unit test does not
re-implement the operation, but verifies the correct behavior by checking mathematical relations for
simple, well-understood cases. For instance, the Laplace operator implemented in ``matvec`` computes the
second derivative of a 1D function in each direction, and should be second order accurate.
A simple example of a unit test is given in test/operations.cpp, where you can add your own as well.
To compile and run the unit tests, use ``make test``. This may be done on the login node when developing on DelftBlue.

2. Compile and run the driver application by typing ``make poisson_cg``. This executable creates a 3D Poisson problem and solves
it using the CG method. A simple CG solver using your operations from task 1 is provided. Add tests to the file ``test/poisson_cg.cpp``
(E.g.: swapping grid dimensions should not change the number of iterations, and the exact solution should be found after ``N`` iterations).

3. Perform weak and strong scaling experiments on a DelftBlue node (using up to 48 cores). Include plots in your report, and interpret them.
When running these jobs, modify the commands in the jobscript ``cg_scaling.slurm`` to your needs, but leave the SBATCH lines unchanged.

4. The bulk-synchronous performance model (BSP) views a program like poisson_cg as a sequence of parallel operations interleaved by communication phases.
It predicts the overall runtime to be the sum of the cost of computation and communication phases. Can you spot opportunities in the CG algorithm to reduce
the number of stages/loops? Implement your own version of the algorithm with fewer loops, and measure if this makes the method faster. Explain your observations
(Hint: we give you the macro ``TIME_SECTION`` to get insight into which operations take how long).
