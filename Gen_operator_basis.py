"""

Generate basis of operators (for local operations T and boxworld process W) for an arbitrary dimension d

It must be run at least once before optimizing the value of a causal inequality (GYNI, LGYNI, or OCB)

"""
import numpy as np
import cvxpy as cp
import mosek
from datetime import datetime
from Basis_fct import oper_basis, W_basis

## Code assumes that all local dimensions are the same
d=2 ## local dimension
N=2 ## number of parties
ns=4 ## number of local spaces, i.e., 2 inputs and 2 outputs
n=ns*N ## total number of spaces 2 inputs and 2 outputs per party

## number of classical inputs and outputs for each operation 
nIN = 2
nOUT = 2

print("Computing basis of operators in dimension", d)

## Define basis of a single local system (O, O', I, or I') as sum-zero operators, except the first one
II = np.ones(d)
Z = np.zeros(d)
Z[0] = 1
BB = [II]

for i in range(d-1):
    aux = np.copy(Z)
    aux[i+1] = -1
    BB += [ aux ]


print("Computing local operations basis")
B = oper_basis(BB, d)

print("Computing W basis")
BW = W_basis(BB, d)

fOP = "op_basis_"+str(d)+"D"
fW = "W_basis_"+str(d)+"D"

print("Saving basis of operators for T and W as", fOP+".npy", fW+".npy")

np.save(fOP, B)
np.save(fW, BW)
