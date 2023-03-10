# Special Topics in CSE: Preconditioned Krylov Methods for High Performance Computing

## Homework 2: CG performance analysis and distributed memory

In the second exercise, we will start by applying simple performance models to determine the
efficiency of your implementation from homework 1.
Ihen, we will extend the implementation to exploit **distributed memory parallelism** using MPI.

The PDE, boundary conditions, discretization and solver are as in [homework 1](https://gitlab.tudelft.nl/dhpc/sticse-hpc/homework1).

## How to complete the homework

### Coding

You will analyze your existing C++/OpenMP implementation, and extend it using the Message Passing Interface (MPI)
to support distributed memory parallelism.

### Report

Write a short report on your findings, in particular answering the questions posed below.
Include the report in PDF format in your submission (see below).

### Working with the repository

- Before you start, fork the repository into your private github account.
- In your cloned repository (on DelftBlue), add the new location as an alternative ``remote`:
    - ``git remote add github https://github.com/<username>/WI4450``
    - update your main branch by ``git checkout main`` and ``git pull github main``
    - update your previous implementation using ``git checkout <netid>`` and ``git merge main``.
      This may or may not cause merge conflicts that you have to resolve (see lecture notes from session 3).
- regularly push your work to the (forked) repository to avoid losing something. The first time you push,
  use the flag ``--set-upstream`` to make it the new default remote location and branch: ``git push --set-upstream github <netid>``.

### Submission

- Add the final report as a PDF file (you may keep a LaTeX file or other source file in the repository, too).
- Push your latest version to forked repository
- Create a merge request, using ``<netid>`` as the source and target branch (on the original repository).

## Your Tasks

1. ... 
2. ...

## Optional bonus task

8. ...
