"""

Collection of basic functions for describing boxworld processes and local operations of arbitrary dimension

"""
import numpy as np
import cvxpy as cp

################################################################################

##########              Basic indices manipulation                    ##########

################################################################################

## Transform list of indices (tensor) into single index (vector)
def ntoone(v, d):
    ind=0
    n=len(v)
    for i in range(n):
        ind += v[i]*d**(n-i-1)

    return ind

## Transform single index (vector) into list of indices (tensor)
def oneton(ind, d, n):
    v = np.zeros(n, dtype=int)
    for i in range(n):
        v[-i-1] = ind % d
        ind = ind // d

    return list(v)

## Fix a label for input output of local operations
def op_lab(a,x):
    return 2*a+x

## Convert vector from new labels to old labels
def convert_labels_back(vec, labels):
    sp_lab  = [ 'AI2', 'AO2', 'AO1', 'AI1',  'BI2', 'BO2', 'BO1', 'BI1'  ]
    old_lab = [ "OA", "O'A", "IA", "I'A", "OB", "O'B" , "IB", "I'B" ]
    new_vec = []
    for j in range(len(old_lab)):
        new_vec += [ vec[labels.index(old_lab[j])] ]

    return new_vec

## Convert vector from in-labels to out-labels
def convert_labels(vec, labin, labout):
    sp_lab  = [ 'AI2', 'AO2', 'AO1', 'AI1',  'BI2', 'BO2', 'BO1', 'BI1'  ]
    old_lab = [ "OA", "O'A", "IA", "I'A", "OB", "O'B" , "IB", "I'B" ]
    new_vec = []
    for j in range(len(old_lab)):
        new_vec += [ vec[labin.index(labout[j])] ]

    return new_vec


################################################################################

##########               Basic tensor operations                      ##########

################################################################################

## General Kronecker product for cvxpy variables with at least one constant
def general_kron(a, b):
    return cp.hstack([ a[k]*b for k in range(a.shape[0]) ] )

## Kronecker product for a list of matrices
def tens_list(mlist):
    ## create tensor product from a list of matrices
    mt = np.array([1])
    for m in mlist:
        mt = np.kron(mt,m)

    return mt

################################################################################

##########            Reduce and replace operations                   ##########

################################################################################

##reduce space sp of tensor T, with n systems and local dimension d
##Label of spaces starting from 1, not 0
def tens_red(T, sp, d, n):
    newT = np.zeros(d**(n-1))
    # sp need offset of 1 to work correctly
    for i in range(len(newT)):
        v = oneton(i, d, n)
        ind = [ ntoone(v[0:sp+1]+[k]+v[sp+1:], d) for k in range(d) ]
        print("ind", ind)
        newT[i] = sum( T[ind[k]] for k in range(d) )

    return newT


##reduce and replace space sp of tensor T, with n systems and local dimension d
##Label of spaces starting from 1, not 0
def red_n_repl(T, sp, d, n):
    newT = np.zeros(d**n)
    # sp need offset of 1 to work correctly
    for i in range(d**(n-1)):
        v = oneton(i, d, n)
        ind = [ ntoone(v[0:sp+1]+[k]+v[sp+1:], d) for k in range(d) ]
        sumi = sum( T[ind[k]] for k in range(d) )
        for k in range(d):
            newT[ind[k]] = sumi

    return newT/d

## RR from a list (e.g., AI1, AI2, AO1)
def red_n_repl_list(T, splist, d, n):
    newT = T.copy()
    for k in range(len(splist)):
        newT = red_n_repl(newT, splist[k], d, n)

    return newT

## RR from an expression (e.g., -AI1AI2 + AO1 ) 
def red_n_repl_expr(T, spexpr, d, n):
    newT = np.zeros(d**n)
    for k in range(len(spexpr)):
        newT += spexpr[k][0]*red_n_repl_list(T, spexpr[k][1], d, n)

    return newT

## RR from a symbolic expression
def red_n_repl_symbolic(T, spsym, d, n, case='W'):
    if n==4:
        spl = [ 'I2', 'O2', 'O1', 'I1' ]
    elif n==8:
        if case == 'W':
            spl = [ 'AI2', 'AO2', 'AO1', 'AI1',  'BI2', 'BO2', 'BO1', 'BI1'  ]
        if case == 'R':
            spl = [ 'AI2', 'AO2', 'AO1', 'AI1', 'tAI2', 'tAO2', 'tAO1', 'tAI1'  ]
            
    else:
        print("Error, wrong number of spaces")
        return 0

    expr = []
    for k in range(len(spsym)):
        coeff = spsym[k][0]
        auxexpr = []
        for i in range(len(spsym[k][1])):
            auxexpr += [ spl.index(spsym[k][1][i]) ]

        expr += [ [coeff, auxexpr] ]

    return red_n_repl_expr(T, expr, d, n)



################################################################################

##########           Generation of operator basis                     ##########

################################################################################

## Generation of constrains from symbolic representation
## Constraints of the form P(X)=X

def symbolic_constraints(case):
    # Constraints for W
    if case == 'W':
        expr = []
        sp = [ 'AI2', 'AO2', 'AO1', 'AI1',  'BI2', 'BO2', 'BO1', 'BI1'  ]
        
        expr += [ [ [ [1], [] ], [ [-1], ['BI1','BI2','BO1','BO2'] ],
                  [ [+1], ['BI1','BI2','BO1','BO2', 'AO2'] ]
                ] ]

        expr += [ [ [ [1], [] ], [ [-1], ['AI1','AI2','AO1','AO2'] ],
                  [ [+1], ['AI1','AI2','AO1','AO2', 'BO2'] ]
                ] ]

        expr += [ [ [ [1], ['AO2'] ],   [ [1], ['BO2'] ], [ [-1], ['AO2', 'BO2']]
                 ] ]


        expr += [ [ [ [1], [] ], [ [-1], ['AI2'] ], [ [1], [ 'AI2', 'AO1'] ] ] ]
        expr += [ [ [ [1], [] ], [ [-1], [ 'BI2'] ], [ [1], [ 'BI2', 'BO1'] ] ] ]


    # Constraints for local operations
    if case == 'T':
        expr = []
        sp = [ 'I2', 'O2', 'O1', 'I1' ]

        expr += [ [ [ [1], [] ], [ [-1], ['O2', 'O1'] ], [ [1], ['I2', 'O2', 'O1', 'I1' ] ] ] ]
        expr += [ [ [ [1], [] ], [ [1], ['I2', 'O2'] ], [ [-1], [ 'O2'] ] ] ]

        
    return expr





## Check operator constraints starting from symbolic expressio
## More precisely, whether P(X)=X
def check_oper_const(X, symexpr, d, ns, case='W', verbose=False):
    ERR = 0.0001
    Xp = X.copy()
    for k in range(len(symexpr)):
        Xp = red_n_repl_symbolic(Xp,symexpr[k], d, ns, case=case)

    if (np.absolute(X - Xp) <= ERR).all():
        return True

    if verbose:
        print("Check orthogonality constraints")
        if (np.absolute(Xp) <= ERR).all():
            print("Element of the orthogonal space")
        else :
            print("Problem with this element")
            print("Original:", list(X))
            print("Projected:", list(Xp))
        
    return False

## Check if operator is 0 after the reduce-and-replace operation
def check_oper_invariance(X, symexpr, d, ns, case='W',verbose=False):
    ERR = 0.0001
    Xp = X.copy()
    for k in range(len(symexpr)):
        Xp = red_n_repl_symbolic(Xp,symexpr[k], d, ns, case)

    if (Xp <= ERR).all():
        print("Element of the orthogonal space")
        return True
        
    return False



## Create basis for local operations
## Take all possible tensor product of local basis and remove those who do not satisfy the constraints
def oper_basis(BB, d):
    # number of input/output spaces
    print("Generating basis for local operations")
    ns = 4
    lb = len(BB)
    B = []
    # remove identity operator because fixed by normalization
    # keep only traceless operators
    symexpr = symbolic_constraints('T')
    for i in range(1,lb**ns):
        print("Checking element", i,"of the basis")
        ol = oneton(i, lb, ns)
        op = tens_list([ BB[ol[k]] for k in range(ns) ])
        if  check_oper_const(op, symexpr, d, ns):
            B += [ op ]

    return B



## Create basis for process tensor (valid only for bipartite case!)
## Same idea as above
def W_basis(BB, d):
    # number of input/output spaces
    print("Generating basis for W")
    ns = 8
    lb = len(BB)
    B = []
    symexpr = symbolic_constraints('W')
    # remove identity operator because fixed by normalization
    # keep only traceless operators
    ## Parallelize this operation!
    for i in range(1,lb**ns):
        print("Checking element", i,"of basis")
        ol = oneton(i, lb, ns)
        op = tens_list([ BB[ol[k]] for k in range(ns) ])
        if  check_oper_const(op, symexpr, d, ns):
            B += [ op ]


    return B
