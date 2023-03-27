# Special Topics in CSE: Preconditioned Krylov Methods for High Performance Computing

## Homework 2: CG performance analysis and optimization

In the second exercise, we will start by applying simple performance models to determine the
efficiency of your implementation from homework 1. Then, we will try to improve the performance
of the previous implementation step-by-step to get close to the predicted timings.

The PDE, boundary conditions, discretization and solver are as in [homework 1](https://gitlab.tudelft.nl/dhpc/sticse-hpc/homework1).

## How to complete the homework

### Coding

You will improve on your existing C++/OpenMP implementation. Testing and benchmarking remain
essential components of the workflow as you adapt your code step-by-step.
For the experiments (other than running tests, which you may do on the login node), submit
a batch job and add the flags ``--exclusive`` and ``--nodes=1``.

### Report

Write a short report on your findings, in particular answering the questions posed below.
You do not need to include source code in the report, this is submitted via the git repository.
Include the report in PDF format in your submission (see below).

### Working with the repository

- If you do not have an account on github, you create one and sign in.
- If you do not have an ssh key registered on github, create one on DelftBlue
  using ``ssh-keygen``, and upload the public key (``.pub`` file) following [these instructions ](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
- Before you start, fork the repository into your private github account.
- In your existing local repository (on DelftBlue), add the new location as an alternative `"remote":
    - ``git remote add github git@github.com:<username>/WI4450``.
    - update your main branch by ``git checkout main`` and ``git pull github main``
    - update your previous implementation using ``git checkout <netid>`` and ``git merge main``.
      This may or may not cause merge conflicts that you have to resolve (see lecture notes from session 3).
- regularly push your work to the (forked) repository to avoid losing something. The first time you push,
  use the flag ``--set-upstream`` to make it the new default remote location and branch: ``git push --set-upstream github <netid>``.

### Submission

- Add the final report as a PDF file (you may keep a LaTeX file or other source file in the repository, too).
- Push your latest version to forked repository
- Create a merge request, using ``<netid>`` as the source and target branch (on the original repository).

## Your tasks

1. If you haven't done so, add a Timer object to ``cg_solve`` and each of your basic operations from homework 1. Run the CG solver for 100 iterations
on a $600^3$ grid on 12 cores and produce a runtime profile, e.g. as a 'pie chart'. What is the approximate size of a vector for this grid size,
and how much memory do you need to request on DelfBlue for the CG solver to run? Does it help to use more aggressive compiler optimization, e.g. ``-O3 -march=native``? If this run takes more than a few minutes, continue with a more feasible grid size and return to this one after you have
optimized your code a bit, see below.
2. Extend the ``Timer`` class to store two additional doubles: the number of floating point operations (flops) performed in the timed section,
and the number of bytes loaded and/or stored. The ``summarize()`` function should print out three additional columns:  
    - the computational intensity of the timed section
    - the average floating point rate achieved (in Gflop/s)
    - the average data bandwidth achieved (in GByte/s)  
Run your benchmark program for the same problem size and 12 cores (with the values inserted in the Timer calls).
What is the limiting hardware factor for each operation based on the roofline diagram from lecture 4? 
**Hint:** for ``apply_stencil3d`` the exact amount of data loaded is unclear due to caching. Here you can start with the most optimistic case (all elements cached after the first time they are accessed).
3. For each operation, determine the applicable peak performance Rmax assuming 12 cores with 2 AVX512 FMA units (see lecture 1). Use the likwid-bench tool to measure the bandwidth on 12 cores (flag ``-w M0:<size>`` where ``<size>`` is the size of a vector). You can get a list of benchmarks it supports using ``-a`` and determine a suitable maximum memory bandwidth for each of your operations by selecting one that has a similar load/store ratio (note that you need to ``module load 2022r2 likwid`` on DelftBlue). Run both the likwid benchmarks and your benchmark program for 1, 2, 4, 6, 8, 10 and 12 threads. Make plots that show
the achieved memory bandwidth for the chosen likwid benchmark and the operation in CG you benchmarked, and note down the absolute efficiency of your code compared to
the roofline model prediction for the case of 12 threads.
4.  Repeat this to create similar graphs for up to 48 threads and report the overall efficiency on a full node. If this is significantly worse than on 12 cores, you may be struggling with the Non-Uniform Memory Architecture (NUMA): 12 cores can access memory **which they  initialized** at the maximum speed (one NUMA domain).
If you go beyond that, you may need to make sure that threads mostly access the memory portions they initialized, by adding the ``schedule(static)`` clause to your ``#pragma omp parallel for`` statements. Also, make sure to set the environment variables ``OMP_PROC_BIND=close`` and ``OMP_PLACES=cores``.
5.  For any operation that performs significantly worse than the roofline prediction on 12 cores (say, less than 50%), try to optimize that operation by  
    - experimenting with compiler flags
    - checking the model assumptions and hardware parameters
    - actually changing the code. For example, if you used if-statements in the apply_stencil3d innermost loop, try to get rid of them. If you have more than one read of the u vector and one write of the v vector because of multiple passes over them, reduce the data traffic. If your code is much faster for certain grid sizes and then suddenly the performance drops as you increase it, try implementing a variant that loops over blocks or try parallelizing the innermost loop instead of the outermost one. Use the 'layer condition' introduced in lecture 6 by Prof. Wellein to guide these optimizations. Document the changes that have a positive effect along with the achieved percentage of the roofline model. And -- obviously -- run your tests after each step to make sure that your code is producing correct results
6.  Finally, what is the total CG runtime on 12 and 48 cores you achieve for the $600^3$ problem, and what is the total runtime predicted by the Roofline model?
