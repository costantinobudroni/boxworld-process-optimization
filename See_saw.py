"""

Collection of basic functions for optimizing boxworld processes and local operations

"""
import numpy as np
import cvxpy as cp
import mosek
from Basis_fct import oneton, general_kron, op_lab

################################################################################

##########           expressions for probabilities                    ##########

################################################################################


## Swap the bits for 00, 01, 10, 11 represented as j=0,...,3, useful for the GYNI expression 
def swapbits(j):
    ## given two bits ab, represented as integer 0,..,3, return ba 
    if j < 0 or j> 3:
        print("Error wrong index j")
        return -1
    
    if j==1:
        return 2
    if j==2:
        return 1
    
    return j


## Generate vector of operators corresponding to each entry of probability vector
def prob_expr(TO, T, case='W'):
    ## p[i][j] = p(a_i b_j| x_i y_j) where i = 0,1, 2, 3
    if not ( case == 'W' or case == 'TA' or case == 'TB'):
        print("ERROR in the generation of the probability expression")
        return 0

    ## case of optimization over W, TA, TB are tensors
    if case == 'W':
        p = [ [np.kron(TO[j], T[k]) for k in range(4)]  for j in range(4) ]

    ## cp.kron works only with a constant as first variable
    ## T = TB, TO = TA (TO is a cvx variable)
    if case == 'TA':
        p = [ [general_kron(TO[j], T[k]) for k in range(4)]  for j in range(4) ]


    ## cp.kron works only with a constant as first variable
    ## T = TA, TO = TB (TO is a cvx variable)
    if case == 'TB':
        p = [ [general_kron(T[j], TO[k]) for k in range(4)]  for j in range(4) ]



    return p

## Same, but for the case of OCB that has extra input on B
def prob_expr_OCB(TO, T, case='W'):
    ## p[i][j] = p(a_i b_j| x_i y_j) where i = 0, 1, 2, 3, j=0,1,2,...,7
    if not ( case == 'W' or case == 'TA' or case == 'TB'):
        print("ERROR in the generation of the probability expression")
        return 0

    ## case of optimization over W, TA, TB are tensors
    if case == 'W':
        p = [ [np.kron(TO[j], T[k]) for k in range(8)]  for j in range(4) ]

    ## cp.kron works only with a constant as first variable
    ## T = TB, TO= TA (TO is a cvx variable)
    if case == 'TA':
        p = [ [general_kron(TO[j], T[k]) for k in range(8)]  for j in range(4) ]


    ## cp.kron works only with a constant as first variable
    ## T = TA, TO= TB (TO is a cvx variable)
    if case == 'TB':
        p = [ [general_kron(T[j], TO[k]) for k in range(8)]  for j in range(4) ]



    return p
    

## Compute different expressions in terms of the tensors TA and TB
def GYNI_expr(TO, T, case):
    ## p[i][j] = p(a_i b_j| x_i y_j) where i = 0,1, 2, 3
    ## convention given by op_lab(a,x)=2*a+x
    p = prob_expr(TO, T, case)
    obj = sum( [ p[swapbits(j)][j] for j in range(4) ])/4
    return obj

def LGYNI_expr(TO, T, case, d=2):
    ## p[i][j] = p(a_i b_j| x_i y_j) where i = 0, 1, 2, 3
    ## convention given by op_lab(a,x)=2*a+x
    p = prob_expr(TO, T, case)
    ##
    ## see convention above, e.g., [2,1] --> p[2][1] = p(10|01)
    sett = [[0, 1], [2, 1], [1, 0], [1, 2], [3, 3]]

    
    obj = 1/4*(1/d**4) + sum([p[sett[j][0]][sett[j][1]]
                              for j in range(len(sett))])/4
    return obj

def OCB_expr(TO, T, case):
    ## p[i][j] = p(a_i b_j| x_i y_j) where i = 0, 1, 2, 3
    ## convention given by op_lab(a,x)=2*a+x
    p = prob_expr_OCB(TO, T, case)
    ##
    ## see convention above, e.g., [2,1] --> p[2][1] = p(10|01)
    obj = 0
    for i in range(4):
        for j in range(8):
            a,x = oneton(i,2,2)
            b, y, yp = oneton(j,2,3)
            if (yp == 0 and a == y) or (yp == 1 and b == x):
                obj += p[i][j]

    return obj/8


################################################################################

##########                See-saw algorithms                          ##########

################################################################################

## See-saw algorithm for the W part (GYNI case)
def see_sawW(BW, TA, TB, n, d, ine='GYNI', full_sol=True):
    lW = len(BW)
    constr = []
    ## Create W tensor as a linear combination of elements of the basis
    ## This guarantees the linear constraints
    cW = cp.Variable((lW))
    W = np.ones(d**n)/(d**(n/2)) + sum( [ cW[i]*BW[i] for i in range(lW) ] )
    constr += [ W >= 0 ]
    if ine == 'GYNI':
        GY = GYNI_expr(TA, TB, 'W')
    elif ine == 'LGYNI':
        GY = LGYNI_expr(TA, TB, 'W', d=d)
    elif ine == 'OCB' :
        GY = OCB_expr(TA, TB, 'W')
    else:
        print("Unknown expression:", ine)
        return 0
    
    objexp =  W.T @ GY
    obj = cp.Maximize(objexp)
    probl = cp.Problem(obj, constr)
    probl.solve(solver=cp.MOSEK, verbose=False)

    if full_sol == True:
        return W.value
    else:
        return obj.value

## See-saw for the local operations parts (GYNI case)
def see_sawT(B, W, T, case, ns, d, ine='GYNI', full_sol=True):
    lo = len(B)
    nIn = 2
    nOut = 2
    constr = []
    ## Create TO tensor, one nonnegative tensor for each input output (e.g., a,x)
    TO = [ cp.Variable(d**ns, nonneg=True) for k in range(nIn*nOut) ]

    ## Create tensors for the linear constraints T_{x=0} and T_{x=1}
    cTO0 = cp.Variable((lo))
    cTO1 = cp.Variable((lo))

    TO0 = np.ones(d**ns)/(d**(ns/2)) + sum( [ cTO0[i]*B[i] for i in range(lo) ] )
    TO1 = np.ones(d**ns)/(d**(ns/2)) + sum( [ cTO1[i]*B[i] for i in range(lo) ] )

    ## Label 00, 01, 10, 11 = 0, 1, 2, 3  a|x ->  2*a+x
    ## TO[0] + TO[2] = T_{0|0} + T_{1|0}
    ## Equate sum to a tensor that satisfies the linear constraints    
    TOl = [TO0, TO1]
    for x in range(2):
        constr += [TO[op_lab(0,x)] + TO[op_lab(1,x)] == TOl[x] ]

    if ine == 'GYNI':
        GY = GYNI_expr(TO, T, case)
    elif ine == 'LGYNI':
        GY = LGYNI_expr(TO, T, case, d=d)

    else:
        print("Unknown expression:", ine)
        return 0

    objexp =  W.T @ GY
    obj = cp.Maximize(objexp)
    probl = cp.Problem(obj, constr)
    probl.solve(solver=cp.MOSEK, verbose=False)

    if full_sol == True:
        return [ TO[k].value for k in range(nIn*nOut) ]

    else:
        return obj.value


## See-saw for the local operations parts (OCB case)
def see_sawT_OCB(B, W, T, case, ns, d, ine='OCB', full_sol=True):
    lo = len(B)
    nIn = 2
    nOut = 2
    constr = []
    if case == 'TA':
        ## Create TO tensor, one nonnegative tensor for each input output (e.g., a,x)
        TO = [ cp.Variable(d**ns, nonneg=True) for k in range(nIn*nOut) ]
        
        cTO0 = cp.Variable((lo))
        cTO1 = cp.Variable((lo))

        TO0 = np.ones(d**ns)/(d**(ns/2)) + sum( [ cTO0[i]*B[i] for i in range(lo) ] )
        TO1 = np.ones(d**ns)/(d**(ns/2)) + sum( [ cTO1[i]*B[i] for i in range(lo) ] )

        ## Label 00, 01, 10, 11 = 0, 1, 2, 3  a|x ->  2*a+x
        ## TO[0] + TO[2] = T_{0|0} + T_{1|0}
        ## Equate sum to a tensor that satisfies the linear constraints    
        TOl = [TO0, TO1]
        for x in range(2):
            constr += [TO[op_lab(0,x)] + TO[op_lab(1,x)] == TOl[x] ]

    elif case == 'TB':
        TO = [ cp.Variable(d**ns, nonneg=True) for k in range(2*nIn*nOut) ]
        
        ## Same reasoning as before, now TB has two inputs: y, y'
        cTO00 = cp.Variable((lo))
        cTO01 = cp.Variable((lo))
        cTO10 = cp.Variable((lo))
        cTO11 = cp.Variable((lo))

        TO00 = np.ones(d**ns)/(d**(ns/2)) + sum( [ cTO00[i]*B[i] for i in range(lo) ] )
        TO01 = np.ones(d**ns)/(d**(ns/2)) + sum( [ cTO01[i]*B[i] for i in range(lo) ] )
        TO10 = np.ones(d**ns)/(d**(ns/2)) + sum( [ cTO10[i]*B[i] for i in range(lo) ] )
        TO11 = np.ones(d**ns)/(d**(ns/2)) + sum( [ cTO11[i]*B[i] for i in range(lo) ] )
        
        TOl = [TO00, TO01, TO10, TO11]
        for y in range(4):
            constr += [TO[y] + TO[y+4] == TOl[y] ]

    if ine == 'OCB' :
        GY = OCB_expr(TO, T, case)

    else:
        print("Unknown expression:", ine)
        return 0

    objexp =  W.T @ GY
    obj = cp.Maximize(objexp)
    probl = cp.Problem(obj, constr)
    probl.solve(solver=cp.MOSEK, verbose=False)

    if full_sol == True:
        return [ TO[k].value for k in range(len(TO)) ]
    else:
        return obj.value
        
################################################################################

##########         Useful local operations and Ws                     ##########

################################################################################


## Generate random local operation, assuming nOUT=2
def gen_rand_oper(B, nIN, d, ns):
    lo = len(B)
    nOUT= 2
    idterm = np.ones(d**ns)/(d**(ns/2))

    ## Generate random element
    Xs = [ sum( [ np.random.rand()*B[i] for i in range(lo) ] ) for k in range(nIN) ]
    for k in range(nIN):
        M = np.amax(Xs[k])
        m = np.amin(Xs[k])
        ## renormalize such that all elements are between -1 and 1
        if np.absolute(m) > np.absolute(M):
            Xs[k] = Xs[k]/np.absolute(m)
        else:
            Xs[k] = Xs[k]/np.absolute(M)

    #print("max and min of Xs:", np.amax(Xs[0]), np.amin(Xs[0])) 
    Ts = [ idterm + Xs[k]/(d**(ns/2)) for k in range(nIN) ]

    T0s = [ Ts[k]*(np.random.rand(d**ns)) for k in range(nIN) ]

    TAs = []
    for j in range(nIN): 
        TAs +=  [ T0s[j] ]

    for j in range(nIN): 
        TAs +=  [ Ts[j]-T0s[j] ]

    return TAs
