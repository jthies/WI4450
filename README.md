# Special Topics in CSE: Preconditioned Krylov Methods for High Performance Computing

## Homework 1: Conjugate Gradient Method with OpenMP

In this exercise, we will exploit **shared memory parallelism** to solve
a standard partial differential equation, the 3D Poisson equation:

```math
- (\frac{\partial^2 u}{\partial x^2} + 
\frac{\partial^2 u}{\partial y^2} + 
\frac{\partial^2 u}{\partial z^2})
= f(x,y,z),
```
on a unit cube $`\Omega = [0\dots 1] \times [0 \dots 1] \times [0\dots 1]`$, subject to Dirichlet boundary conditions

- u(x,y,z)=0 if x=0, x=1, y=0, y=1, or z=1,
- u(x,y,z)=g(x,y) if z=0.

The PDE is discretized using second order finite diferences, and solved via the Conjugate Gradient method.

## How to complete the homework

### Coding

- The implementation will be done in (relatively basic/C-like) C++.
- Places where you have to implement something are marked in the code skeleton by an ellipse ``[...]``.
- Useful comments in your code are part of the assessment!

### Report

Write a short report on your findings, in particular answering the questions posed below.
Include the report in PDF format in your submission (see below).

### Working with the repository

- Clone the repository on DelftBlue using ``git clone https://gitlab.tudelft.nl/<netid>/homework1``.
- Create a branch ``<netid>`` and work on this branch (or simply ``git checkout <netid>`` if it already exists).
- regularly push your work to the repository to avoid losing something

### Submission

- Add the final report as a PDF file (you may keep a LaTeX file or other source file in the repository, too).
- Push your latest version of the ``<netid>`` branch
- Create a merge request, using ``<netid>`` as the source and ``<netid>-submission`` as the target branch. If the latter
branch does not have this branch for your NetID, create an issue to request it
in the [issue tracker](https://gitlab.tudelft.nl/dhpc/sticse-hpc/homework1/issues/).

## Your Tasks

1. implement the missing functions ``init``, ``axpby``, ``dot``,and ``apply_stencil3d`` using standard sequential for-loops.
   The specification of these functions can be found in the file ``operations.hpp``.
   Implement suitable unit tests to verify they work as expected. Remember that a good unit test does not
   re-implement the operation, but verifies the correct behavior by checking mathematical relations for
   simple, well-understood cases. For instance, the Laplace operator implemented in ``matvec`` computes the
   second derivative of a 1D function in each direction, and should be second order accurate.
   Some simple examples of a unit test is given in ``test_operations.cpp``, where you can add your own as well.
   To compile and run the unit tests, use ``make test``. This may be done on the login node when developing on DelftBlue.
   **Note:** you can implement the matrix-vector product with a 7-point stencil in any way you like (matrix-free, by storing
   diagonals, etc.). Start with a simple implementation and create sufficient unit tests so that you can then refine it.
   The tricky part here are the boundary conditions.

2. Complete the Conjugate Gradient method in ``cg_solver.cpp`` by calling ythe previously implemented functions in the
   locations indicated by an ellipse (``[...]``). Compile and run the driver application by typing ``make``. This executable creates a 3D Poisson problem and solves
   it using the CG method. Add tests to a new file ``test_cg_solver.cpp`` and modify the makefile so that it is compiled into the ``run_tests.x`` executable.
   ``n`` iterations).

3. Parallelize your basic opera5ions from step 1 using OpenMP.  For the ``apply_stencil3d`` function, parallelize the outer loop only.
   Rerun your tests using ``make test`` to verify that everything still works.

4. Add a new main program ``main_benchmarks.cpp`` and update the ``Makefile`` to compile an executable ``main_benchmarks.x``.
   You may find the ``Timer`` object useful, defined in ``timer.hpp``.
   For the individual operations from task 1,
   perform weak and strong scaling experiments on a DelftBlue node (using up to 48 cores). Include plots in your report, and interpret them.
   To run these jobs, modify the commands in the jobscript ``run_cg_poisson.slurm`` to your needs, but leave the SBATCH lines unchanged.
   How many cores are optimal for the ``main_cg_poisson.x`` application on a $512^3$ grid?

5. For the ``apply_stencil3d`` operation, try interchanging the three nested loops, and run the code sequentially (set ``OMP_NUM_THREADS=1``).
   Which loop order is best, and why? In case you stored the matrix entries in task 1, change the order in which they are accessed in ``apply_stencil3d``.

6. Run the main_cg_poisson.x driver. Do you observe a performance difference when solving a $`1024 \times 128 \times 128`$ or a $`128 \times 128 \times 1024`$ problem on 8 cores? Add the ``collapse(3)`` clause to the OpenMP pragma line in ``apply_stencil3d`` and run again.
Explain your observations.

7. The bulk-synchronous performance model (BSP) views a program like poisson_cg as a sequence of parallel operations interleaved by communication phases.
It predicts the overall runtime to be the sum of the cost of computation and communication phases. Can you spot opportunities in the CG algorithm to reduce
the number of stages/loops? Implement your own version of the algorithm with fewer loops, and measure if this makes the method faster. Explain your observations.

## Optional bonus task

8. Make your basic operations run on the GPU using OpenMP ``target`` directives or C++20 (see [this repository](https://gitlab.tudelft.nl/dhpc/training/cxx-examples) for C++20 examples). Use the nvhpc compilers and the DelftBlue GPU nodes, as illustrated in that repo. Are all your tests still passing? Is the CG solver working correctly? And how fast is it on a V100s GPU compared to a standard compute node? A ``data`` statement in the outer CG scope can be used to make this run more efficient...
