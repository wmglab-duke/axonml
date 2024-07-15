"""Data collection configuration."""

# name of dataset file
filename = "train"

# size of dataset
n_batches = 64
batch_size = 32
axons_per_minibatch = 32

# sim params
tstop = 5.0
n_node_record = 53
n_contacts = 6

# datagenerator metadata
round = 0  # round of data collection

# datagenerator params
diameter = 5.7
use_dr = True
min_diam = 5.7
max_diam = 14.0

max_amp = 0.2
max_pw = 2.0
max_delay = 2.0

# record what?
record_e = True
record_v = True
