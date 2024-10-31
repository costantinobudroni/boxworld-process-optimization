"""
Boxworld Process Optimization

This package is designed to perform numerical optimization for Boxworld process correlations.
In particular, it optimizes the value of GYNI, LGYNI, and OCB inequality via seesaw algorithms
based on linear programming.

Requirements: 
- numpy: for numerical operations 
- cvxpy: for convex optimization
- mosek: as LP solver
- joblib: for parallelization

Usage:
Import the package and use the main functions provided to set up your problem and perform 
the optimization. See the documentation and examples for detailed usage instructions.

Author: Costantino Budroni
Date: Oct. 31 2024
"""
import numpy as np
import cvxpy as cp
import mosek
import sys
from datetime import datetime
from joblib import Parallel, delayed
from Basis_fct import *
from See_saw import *

## Code assumes that all local dimensions are the same
d=2 ## local dimension
N=2 ## number of parties
ns=4 ## number of local spaces, i.e., 2 inputs and 2 outputs
n=ns*N ## total number of spaces 2 inputs and 2 outputs per party

## number of classical inputs and outputs for each operation 
nIN = 2
nOUT = 2

## Fix parameters for the optimization (initial value, number of iterations for seesaw,
##  number of runs with random initial conditions, number of cores for parallel computation)
max_found = 0
num_iter = 10
num_runs = 20
num_cores = 8


## Verify and save solution
verify_solution = True
save_solution = True
Wsol_filename = "W_sol_LGYNI"+str(d)+"D"
TAsol_filename = "TA_sol_LGYNI"+str(d)+"D"
TBsol_filename = "TB_sol_LGYNI"+str(d)+"D"


## Load local basis of operators
fOP = "op_basis_"+str(d)+"D.npy"
fW = "W_basis_"+str(d)+"D.npy"
try :
    B = np.load(fOP)
    BW = np.load(fW)

except OSError as e:
    print(e)
    print("Operator basis not found. Try to run 'Gen_operator_basis.py' to generate it.")
    sys.exit()


## Define seesaw algorithm starting from random initial local operations
def see_saw_step(num_iter):
    TAs = gen_rand_oper(B, nIN, d, ns)
    TBs = gen_rand_oper(B, nIN, d, ns)

    
    for _ in range(num_iter):
        Ws = see_sawW(BW, TAs, TBs, n, d, ine='LGYNI')
        TAs = see_sawT(B, Ws, TBs, 'TA', ns, d, ine='LGYNI')
        TBs = see_sawT(B, Ws, TAs, 'TB', ns, d, ine='LGYNI')


    GY = LGYNI_expr(TAs, TBs, 'W')
    obj_val = Ws.T @ GY

    return obj_val, Ws, TAs, TBs
    

# get the running time 
start_time = datetime.now()

print("Optimzing LGYNI inequality for boxworld processes of dim", d)

for i in range(num_runs):
    ret_v = Parallel(n_jobs=num_cores)(delayed(see_saw_step)(num_iter)  for k in range(num_cores))

    value_v = np.array([ ret_v[i][0] for i in range(len(ret_v))])

    obj_val = max(value_v)

    if obj_val > max_found:
        pos = np.argmax(value_v)
        print("new maximum found:", obj_val)
        max_found = obj_val
        Wsol = ret_v[pos][1]
        TAsol = ret_v[pos][2]
        TBsol = ret_v[pos][3]

time_elapsed = datetime.now() - start_time

print("Time elapsed in hh:mm:ss =", time_elapsed)

if verify_solution:
    ## Generate constraints for W in symbolic form
    symexpr= symbolic_constraints('W')

    print("Conditions for W:", check_oper_const(Wsol, symexpr, d, n, verbose=True))

    print("Normalization of W:", sum(Wsol))

    ## Generate constraints for T in symbolic form
    symexpr= symbolic_constraints('T')

    ## Check constraints for T_x, x=1,2
    ## Label of operations 00, 01, 10, 11 = 0, 1, 2, 3  a|x ->  2*a+x
    for x in range(nIN):
        print("Conditions for TA[%d]:" %x, check_oper_const(TAsol[op_lab(0,x)]+TAsol[op_lab(1,x)], symexpr, d, ns, verbose=True))
        print("Normalization of TA[%d]:" %x, sum(TAsol[op_lab(0,x)]+TAsol[op_lab(1,x)]))

    for x in range(nIN):
        print("Conditions for TB[%d]:" %x, check_oper_const(TBsol[op_lab(0,x)]+TBsol[op_lab(1,x)], symexpr, d, ns, verbose=True))
        print("Normalization of TB[%d]:" %x, sum(TBsol[op_lab(0,x)]+TBsol[op_lab(1,x)]))


if save_solution:
    print("Saving solution in ", Wsol_filename+".txt", TAsol_filename+".txt", TBsol_filename+".txt")
    head = '# W process, label of spaces [ "OA", "O\'A", "IA", "I\'A", "OB", "O\'B" , "IB", "I\'B" ]'
    np.savetxt(Wsol_filename, Wsol, header=head, fmt='%.7e')
    head = "# Each row represents T[a,x] (ordering 2*a+x), label of spaces as O, O', I, I'"
    np.savetxt(TAsol_filename, TAsol, header=head, fmt='%.7e')
    head = "# Each row represents T[b,y] (ordering 2*b+y), label of spaces as O, O', I, I'"
    np.savetxt(TBsol_filename, TBsol, header=head, fmt='%.7e')
