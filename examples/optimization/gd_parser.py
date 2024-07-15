import argparse

import gd_config as config

parser = argparse.ArgumentParser()
parser.add_argument("--validate", action="store_true")
parser.add_argument("--model", default=config.model)

# samples
parser.add_argument("-s", "--samples", nargs="+", default=config.samples)

# GD params
parser.add_argument("--lr", type=float, default=config.lr)
parser.add_argument("--lr-decay", type=float, default=config.lr_decay)
parser.add_argument("--n-steps", type=int, default=config.n_steps)

# fiber params
parser.add_argument("--diameter", type=float, default=config.diameter)
parser.add_argument("--nodes", type=int, default=config.nodes)
parser.add_argument("--nc", type=int, default=config.nc)

# parse
args = parser.parse_args()
