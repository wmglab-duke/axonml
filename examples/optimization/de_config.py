"""DE + Surrogate config"""


# DE params
strategy = "best1bin"
mutation = 0.8
recombination = 1.0
parameter_adaptation = 0.1
popsize = 300
maxiter = 500
init = "rlhs"
seed = 12345

# models params
diameter = 5.7
nodes = 101
nc = 6

# search bounds
lb = -0.3
ub = 0.3
