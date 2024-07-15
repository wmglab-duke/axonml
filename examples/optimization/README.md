# Optimization

## Data

Optimization problems are specified under individual subdirectories in `./samples/`. For example, the contents of directory P5 are:

```bash
./samples/P5
├── 0.npy
├── 1.npy
├── 2.npy
├── 3.npy
├── 4.npy
├── 5.npy
├── fiber_z.npy
├── fibers_xy.npy
├── target.npy
└── weights.npy
```
All optimization problems must include:
- Arrays `{i}.npy` for each contact in the electrode, where `i` is the contact index (in the case of P3 there are 6 corresponding to the 6 contacts of the ImThera cuff). Each must be a 2D array with dimensions `n_axons, n_samples`, where `n_axons` is the total number of axons in the model and `n_samples` are the number of longitudinal locations along the length of the nerve at which the extracellular potential has been sampled.
- Array `fiber_z.npy`. This is a 1D array length `n_samples`, coresponding to the longitudinal coordinates (in $\mu m$) at which the field has been sampled.
- Array `target.npy`. This is a 1D boolean array length `n_axons` indicating which axons are targeted for activation (1 / True) and which aren't (0 / False).
- Array `weights.npy`. This is a 1D boolean array length `n_axons` indicating the proportion of fascicular area that each axon represents. The sum over all elements in `weights.npy` must be 1.


## Differential Evolution (DE)

Run for a given sample:

```bash
> python differentialevolution.py --sample $SAMPLE
```

To validate with NEURON, use the `--validate` command-line flag.

### Parameters
Set these in `de_config.py`.

|Variable|Description|
|---|---|
|`strategy`|DE mutation strategy. Choose from: `'best1bin', 'randtobest1bin', 'currenttobest1bin', 'currenttopbestbin', 'best2bin', 'rand2bin', 'rand1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp', 'currenttopbestexp', 'best2exp', 'rand2exp'` **Default: best1bin**|
|`mutation`|Mutation rate. **Default: 0.8**|
|`recombination`|Crossover rate. **Default: 1.0**|
|`parameter_adaptation`|Parameter adaptation rate[^1]. **Default: 0.1**|
|`popsize`|Size of DE population of candidate solutions. **Default: 300**|
|`maxiter`|Maximum number of DE iterations. **Default: 500**|
|`init`|Population initialization strategy. Options: `'random', 'rlhs', 'mlhs'`. **Default: rlhs**|
|`seed`|Random seed. NOTE: random seed will only be applied if the script is run with the `--seed` command line argument. **Default: 12345**|
|`diameter`|Fiber diameter ($\mu m$). **Default: 5.7**|
|`nodes`|# nodes of Ranvier per fiber. **Default: 101**|
|`nc`|# of electrode contacts. **Default: 6** (all provided examples are for the 6 contact ImThera cuff).|
|`lb`|Lower bound for search (mA). **Default: -0.3**|
|`ub`|Upper bound for search (mA). **Default: 0.3**|

All configuration parameters can be set via the commandline. E.g., to run for sample P5 with parameter adaptation rate `0.3`:

```bash
> python differentialevolution.py --sample P5 --parameter-adptation 0.3
```

## Gradient Descent (GD)

Run for biphasic rectangular waveforms:

```bash
> python gd_biphasic.py
```

or arbitrary waveforms:

```bash
> python gd_arbitrary.py
```

To validate with NEURON, use the `--validate` command-line flag.


### Parameters
Set these in `gd_config.py`.

|Variable|Description|
|---|---|
|`model`|Name of saved model. **Default: 'MRG'**|
|`samples`|Which optimization problems to attempt solving, supplied as a list, e.g. ['H5', 'P5']. **Default: 'all'** (all optimization problems in `'./samples/` will be attempted simultaneously).|
|`lr`|Learning rate for Ranger optimizer. **Default: 2.0**|
|`lr_decay`|Learning rate decay. **Default: 0.6**|
|`n_steps`|# GD steps. **Default: 200**|
|`diameter`|Fiber diameter ($\mu m$). **Default: 5.7**|
|`nodes`|# nodes of Ranvier per fiber. **Default: 101**|
|`nc`|# of electrode contacts. **Default: 6** (all provided examples are for the 6 contact ImThera cuff).|

All configuration parameters can be set via the commandline. E.g., to run for biphasic waveforms for sample H5 and P5 with learning rate decay `0.9`:

```bash
> python gd_biphasic.py --samples H5 P5 --lr-decay 0.9
```

[^1]: Jingqiao Zhang, and A.C. Sanderson. 2009. “JADE: Adaptive Differential Evolution With Optional External Archive.” IEEE Transactions on Evolutionary Computation 13 (5): 945–58. https://doi.org/10.1109/TEVC.2009.2014613.
