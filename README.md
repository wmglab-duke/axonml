<div align="center">
  <img src="docs/banner.png">
</div>

***

[![DOI](https://zenodo.org/badge/829092590.svg)](https://zenodo.org/doi/10.5281/zenodo.12752386)

> Icon courtesy of DALL·E 3, with prompt "A differentiable massively parallel GPU-based model of neural fiber dynamics for prediction and optimization of extracellular electrical stimulation."

*Implement and train high-throughput GPU-compatible neural fiber models.*

## ❗Requirements

### Hardware requirements
`axonml` requires a standard computer with an NVIDIA GPU (we ran all our simulations using an RTX A5000) and enough RAM to support the in-memory operations (loading data for training, constructing input voltage arrays, etc.).

### OS requirements
`axonml` has been tested on Windows 11 under WSL2 (Ubuntu 22.04) and Linux (AlmaLinux v9.3, binary-compatible with Red Hat Enterprise Linux).

### Python dependencies
`axonml` requires Python 3.9+ (tested with 3.9, 3.10) and PyTorch 2.0+ with GPU support (tested with PyTorch 2.0.0 & CUDA 11.7). See `requirements.txt` for additional package dependencies.


## 🖥️ Installation

> [!TIP]
> We recommend using `conda` to manage your python environment. If you have `conda` installed, you may wish to set up a new environment: `conda create -n axonml python=3.10`. Be sure to activate your new environment (`conda activate axonml`) before following the installation instructions or running code.

1.  Install PyTorch (with GPU support, check which CUDA version you have / is compatible with your GPU) - follow the installation instructions [on the PyTorch homepage](https://pytorch.org/).

> [!IMPORTANT]
> Differences in GPU hardware and PyTorch / CUDA version may affect the performance and numerical outcomes of simulations / optimizations. All presented results use PyTorch 2.0.0 and CUDA 11.7.

2.  Clone this repository.

3.  Install requirements : `pip install -r requirements.txt`

4.  Add the cloned `axonml` directory to your `PYTHONPATH`.

🥳 You're all set! 

> [!NOTE]
> Installation of all dependencies should not take more time than a couple of minutes. Installation of `axonml` itself takes only the time required to clone the repository.

> [!IMPORTANT]
> The [`cajal`](https://github.com/minhajh/cajal) package is required to run some of the provided examples - to execute NEURON simulations, use the data generation algorithms, run high-throughput surrogate simulations of kHz stimulation, and perform stimulus optimization (using Differential Evolution[^1] or Gradient Descent) for selective activation. Follow the installation instructions [in that repository](https://github.com/minhajh/cajal) (however do not create a separate `conda` environment for `cajal` - install all dependencies into `axonml`).

## :rocket: Training a model

By default, `axonml` trains approximations to the MRG[^2] myelinated fiber model. Check `axonml/models/README.md` for instructions on how to implement approximations of other fiber models. 

Training configurations can be modified by changing the relevant values in `config.py`:

|Variable|Description|
|---|---|
|`model`|Surrogate fiber model `class` to train. **Default: `axonml.models.SMF`**.|
|`cuda`|Whether to use GPU. **Default: True if GPU is available, False otherwise**.|
|`fp32`|Whether to use single-precision floating point arithmetic. If not, double precision is used. **Default: False**.|
|`nodes`|Nodes of Ranvier per axon. **Default: 53**.|
|`dt`|Simulation timestep [ms]. **Default: 0.005**.|
|`train_dset`|Path to training dataset.|
|`valid_dset`|Path to validation dataset.|
|`states`|The state variables in the training / validations dataset(s) and the order in which they will be concatenated. Must agree with the order in which states are concatenated when recorded from the surrogate model. **Default: ['m', 'h', 'p', 's', 'v']**.|
|`epochs`|Number of training epochs. **Default: 5**.|
|`truncation_length`|Sequence length over which to perform truncated backpropagation through time. **Default: 50**.|
|`lr`|Adam optimizer learning rate. **Default: $3\times10^{-5}$**.|
|`grad_accumulation`|Whether to use gradient accumulation over disjoint chunks in truncated backpropagation through time. **Default: False**.|
|`train_n_idx`|Total # training set batches to use per training epoch (see `./axonml/data/generate_data.py` - `n_batches`). **Default: 64**.|
|`val_n_idx`| # validation set batches to use per round of validation. **Default: 8**.|
|`train_chunk_size`| # training set batches to use per gradient-descent step. (Should be a factor of `train_n_idx`). **Default: 2**.|
|`val_chunk_size`| # validation set batches to use per validation step. **Default: 8**.|
|`sampling`|Whether to downsample training set in time, and by how much (sample every `sampling` timesteps). **Default: None**.|
|`postfix`|List of model parameters to display in progressbar as training progresses. **Default: None**.|
|`save_every`|Save model parameters every `save_every` minibatch iterations. **Default: 32**.|
|`save_dir`|Location into which to save model parameters. **Default: `./checkpoints/`**.|

Once you've set variables appropriately in `config.py`, you can initiate training:

```bash
(base) foo@bar : ~ $ cd /path/to/cloned/repository
(base) foo@bar : /path/to/cloned/repository $ conda activate axonml
(axonml) foo@bar : /path/to/cloned/repository $ python train.py
```
We provide small example training and validation sets for you to explore running the training algorithm. Beware that running the training script on these datasets will only execute the first step of each training / validation epoch.

## 🗄️ Loading a model
Trained `axonml.models.Axon` models can be loaded using the `load` method. We have included a trained version of the MRG fiber (the 'surrogate myelinated fiber', S-MF, pronounced 'smurf'):

```python
# import surrogate myelinated fiber class
from axonml.models import SMF

# instantiate model and load pre-trained parameters
mrg = SMF().cuda().load('MRG')

# ... use mrg for thresholding, modeling, stimulus optimization, etc.
```

To load from checkpoints generated by `train.py`:

```python
# import pytorch
import torch

# import surrogate myelinated fiber class
from axonml.models import SMF

# load checkpoint
checkpoint_path = '/path/to/checkpoint'
checkpoint_params = torch.load(checkpoint_path)['model_state_dict']

# instantiate model and load trained parameters
mrg = SMF().cuda().load(checkpoint_params)

... etc.
```

## 🤖 Running simulations

You need to supply an extracellular potential boundary condition to run simulations. This must be a `torch.Tensor` of shape `(n_timesteps, n_axons, 1, n_nodes)`; for example, if your goal is to simulate the response of 50 axons each with 51 nodes of Ranvier to extracellular stimulation over 5 ms with a timestep of 0.005 ms, the input `ve` should be shape `(1000, 50, 1, 51)`. `ve[100, 0, 0, 4]` is then $V_e$ in mV at node 5 for the 1st axon you're simulating at time t=0.5 ms.

You must also specify the diameters of the fibers being simulated; this must be a 1D `torch.Tensor` of shape `n_axons`.

Optionally, you can supply an array representing intracellular current simulation (in mA), e.g. to simulate synaptic input; this must also be a `torch.Tensor` of shape `(n_timesteps, n_axons, 1, n_nodes)`.

You can specify `dt`; by default, this is 0.005 ms. You can also set `dt` globally using the Backend.

You can then run simulations:

```python
# set dt globally
from axonml import Backend as A
A.dt = 0.001

n_axons, n_nodes = 50, 51
ve = build_ve(50, 51)             # implement this function yourself
intra = build_intra()             # or None
diams = 5.7 * torch.ones(n_axons) # we're simulating 5.7 um fibers

model.run(ve=ve, diameters=diams, intra=intra)
```

You can continue running from where you left off, e.g. run without any extracellular stim for a further 1 ms:

```python
ve = torch.zeros(1000, n_axons, 1, n_nodes)
model.run(ve, diameters=diams)
```

or you can reinitialize and run from steady-state:
```python
ve = torch.zeros(1000, n_axons, 1, n_nodes)
model.run(ve, diameters=diams, reinit=True)
```

### Callbacks
To extract information from these simulations, use `Callback`s. We have implemented `Recorder`, `Active`, `APCount`, and `Raster`.

**`Recorder`** records the system state at every timestep of simulation:
```python
from axonml.models.callbacks import Recorder
rec = Recorder()
model.run(ve, diams, callbacks=[rec])

record = rec.stack()
```

**`Active`** checks if any action potentials have occurred. You can specify threshold (by default 0 mV), time after which to start checking for activation (by default 0 ms), and node indices to monitor (by default [5, -5]). For example, to check if any APs exceeding $V_m$ = 20 mV arrived 10 internodal lengths from the proximal end of each fiber at least 5 ms after t=0 ms :
```python
from axonml.models.callbacks import Active

active = Active(threshold=20.0, t_start_check=5, node_check=[10])
model.run(ve, diams, callbacks=[active])
print(active.record)
```

**APCount** counts the number of action potentials and **Raster** records when they occurred. For both of these, you can also specify threshold (by default 0 mV), time after which to start checking (by default 0 ms), and node indices to monitor (by default [5, -5]).

You can use multiple callbacks at once.

## 🌍 Other functionality
Further instructions and examples of how to estimate thresholds, perform selective stimulus parameter optimization, and run other simulations can be found in `./examples`.

## 📜 License
The copyrights of this software are owned by Duke University. As such, it is offered under a custom license (see LICENSE.md) whereby:

1. DUKE grants YOU a royalty-free, non-transferable, non-exclusive, worldwide license under its copyright to use, reproduce, modify, publicly display, and perform the PROGRAM solely for non-commercial research and/or academic testing purposes.  

2. In order to obtain any further license rights, including the right to use the PROGRAM, any modifications or derivatives made by YOU, and/or PATENT RIGHTS for commercial purposes, (including using modifications as part of an industrially sponsored research project), YOU must contact DUKE’s Office for Translation and Commercialization (Digital Innovations Team) about additional commercial license agreements.

Please note that this software is distributed AS IS, WITHOUT ANY WARRANTY; and without the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

[^1]: Storn, Rainer, and Kenneth Price. 1997. “Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces.” Journal of Global Optimization 11 (4): 341–59. https://doi.org/10.1023/A:1008202821328.

[^2]: McIntyre, Cameron C., Andrew G. Richardson, and Warren M. Grill. 2002. “Modeling the Excitability of Mammalian Nerve Fibers: Influence of Afterpotentials on the Recovery Cycle.” Journal of Neurophysiology 87 (2): 995–1006. https://doi.org/10.1152/jn.00353.2001.
