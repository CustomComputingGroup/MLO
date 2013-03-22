# solve_chol - solve linear equations from the Cholesky factorization.
# Solve A*X = B for X, where A is square, symmetric, positive definite. The
# input to the function is R the Cholesky decomposition of A and the matrix B.
# Example: X = solve_chol(chol(A),B);
#
# NOTE: The program code is written in the C language for efficiency and is
# contained in the file solve_chol.c, and should be compiled using matlabs mex
# facility. However, this file also contains a (less efficient) matlab
# implementation, supplied only as a help to people unfamiliar with mex. If
# the C code has been properly compiled and is avaiable, it automatically
# takes precendence over the matlab code in this file.
#
# Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2010-09-18.

import numpy as np

def solve_chol(L, B):
    try:
        assert(L.shape[0] == L.shape[1] and L.shape[0] == B.shape[0])
    except AssertionError:
        raise Exception('Wrong sizes of matrix arguments in solve_chol.py');

    X = np.linalg.solve(L,np.linalg.solve(L.T,B))
    return X
