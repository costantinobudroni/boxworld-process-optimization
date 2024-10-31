# boxworld-process-optimization
Code for optimization of boxworld process correlations. For more details see the arXiv preprint: arXiv:2410.XXXX 
## Usage Guide for Boxworld Process Optimization Package

To use this package, follow these steps:

### Installation
First, make sure you have the following Python packages installed:
- `numpy`: for numerical operations
- `cvxpy`: for convex optimization
- `mosek`: for LP solver
- `joblib`: for parallelization

### Optimization
To optimize the inequalities for boxworld processes of local dimension d=2, just run Opt_BW_XXX_paral.py, where XXX=GYNI, LGYNI, or OCB.
Modify the optimization parameters (number of iterations, runs, cores, etc.) directly on the corresponding file.
For higher dimension, it is necessary to first generate the corresponding basis of operators with Gen_operator_basis.py
