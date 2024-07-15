import torch
from tqdm import tqdm
import numpy as np

from cajal.nrn.sources import FEMInterpolate1D

from axonml.models import SMF
from axonml.models.callbacks import APCount


torch.set_default_dtype(torch.float32)


def deltax(diam):
    return -8.215284e00 * diam**2 + 2.724201e02 * diam + -7.802411e02


s_idx = 0  # deliver stimulation from contact 0
nodes = 101  # nodes of Ranvier per fiber


# -- load fields --
FIELD_DATA = [np.load(f"./fields/{i}.npy") for i in range(6)]
field_x = np.load("./fields/fiber_zs.npy")


# -- generate field bases --
# machinery
interpolator = FEMInterpolate1D(FIELD_DATA[s_idx] * 1000, field_x)


def make_ve_at_nodes(diameter, a_idx, nodes=nodes, offset=37500):
    start = (deltax(diameter) * (nodes - 1)) / 2
    dx = np.linspace(-start, start, nodes)
    interp_at = np.linspace(-start, start, nodes) + offset
    b = interpolator._interpolate(interpolator.get(a_idx), interpolator.x[0], interp_at)
    return b


diam_amp_dict = {
    5.7: np.arange(0, 10, 0.01),
    8.7: np.arange(0, 5, 0.01),
    14.0: np.arange(0, 2, 0.01),
}


a_indices = [0, 10, 15, 25, 32, 33, 45]


field_stack = []
diams = []
splits = []

# -- construct bases --
for diam, amps in diam_amp_dict.items():
    split_length = 0
    for a_idx in a_indices:
        b = make_ve_at_nodes(diam, a_idx)
        stack = np.einsum("i,j->ij", amps, b)
        field_stack.append(stack)
        diams.append([diam] * len(stack))
        split_length += len(stack)
    splits.append(split_length)

splits = np.cumsum(splits)


field_stack = np.vstack(field_stack)
diam = np.concatenate(diams)


# waveforms

from functools import partial


def waveform(stim, **kwargs):
    return partial(stim, **kwargs)


def sine(t, amp, freq, delay):
    sig = np.sin(2 * np.pi * freq * (t - delay))
    sig[t < delay] = 0
    return sig


def rect(T):
    """create a centered rectangular pulse of width $T"""
    return lambda t: (0 <= t) & (t < T)


def pulse_train(t, at, shape):
    """create a train of pulses over $t at times $at and shape $shape"""
    return np.sum(shape(t - at[:, np.newaxis]), axis=0)


# code to run and count APs

count = APCount(node_check=[-5], threshold=-20.0, t_start_check=50, dt=0.001)


def longrun(
    model,
    tstop,
    dt,
    stims,
    field_stack,
    diams,
    chunks=20,
    count_only=True,
    with_intra=True,
    warmup=True,
):
    field_stack = torch.tensor(field_stack, device="cuda").float().unsqueeze(1)
    diams_gpu = torch.tensor(diams, device="cuda")

    if warmup:
        print("warming up...")

        ve = torch.rand(5, len(field_stack) * len(stims), 1, nodes).float().cuda()
        intra = torch.zeros_like(ve).cuda()
        dg = torch.rand(len(field_stack) * len(stims)).cuda()

        for _ in range(3):
            with torch.no_grad():
                out = model(ve, dg, intra=intra, dt=dt)

    t_vec = np.arange(0, tstop, dt)
    views = np.array_split(t_vec, chunks)

    returns = []
    tstart = 0.0

    for i, t_chunk in enumerate(tqdm(views, desc="Running")):
        input_ve = []
        input_intra = []
        input_diams = []

        for stim in stims:
            tc = stim(t=t_chunk)
            t_course = torch.tensor(tc, device="cuda")
            ve = torch.einsum("i, jkl -> ijkl", t_course, field_stack).float()
            input_ve.append(ve)
            input_diams.append(diams_gpu)
            if with_intra:
                intra = torch.zeros_like(ve)
                i_stim = pulse_train(t_chunk, np.array([50, 60, 70, 80, 90]), rect(0.1))
                intra[:, :, :, 5] = 2e-6 * torch.tensor(i_stim)[:, None, None]
                input_intra.append(intra)

        input_ve = torch.cat(input_ve, 1)
        input_diams = torch.cat(input_diams)
        if input_intra:
            input_intra = torch.cat(input_intra, 1)
        else:
            input_intra = None

        with torch.no_grad():
            reinit = False
            if i == 0:
                reinit = True
            _ = model(
                input_ve,
                input_diams,
                intra=input_intra,
                dt=dt,
                callbacks=[count],
                reinit=reinit,
            )
        tstart = t_chunk[-1]

    return 0


# run

mrg = SMF(handle_nan=True).cuda().load("MRG")

frequencies = [1, 2, 5, 10]
stims = [waveform(sine, amp=1.0, freq=freq, delay=0.5) for freq in frequencies]

count.reset()
_ = longrun(mrg, 100, 0.001, stims, field_stack, diam, chunks=500, warmup=True)
all_n = count.record.cpu().numpy()


# visualize

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TKAgg")

data = {
    "frequency": [],
    "a_idx": [],
    "amplitude": [],
    "diameter": [],
    "# APs": [],
    "method": [],
}

all_n = np.split(all_n, len(frequencies))
all_diams = [5.7, 8.7, 14.0]

for frequency, n in zip(frequencies, all_n):
    n_by_diam = np.split(n, splits[:2])
    for d, n_d in zip(all_diams, n_by_diam):
        n_by_a_idx = np.split(n_d, len(a_indices))
        for a_idx, n_a in zip(a_indices, n_by_a_idx):
            amps = diam_amp_dict[d]
            n_nrn = np.load(
                f"./ground_truth/{frequency}khz_{d}um_2_3_0_{a_idx}.npy"
            ).flatten()
            for amp, n_ in zip(amps, n_a):
                data["frequency"].append(frequency)
                data["a_idx"].append(a_idx)
                data["amplitude"].append(amp)
                data["diameter"].append(d)
                (data["# APs"].append(n_[0]),)
                data["method"].append("Surrogate")
            for amp, n_ in zip(amps, n_nrn):
                data["frequency"].append(frequency)
                data["a_idx"].append(a_idx)
                data["amplitude"].append(amp)
                data["diameter"].append(d)
                data["# APs"].append(n_)
                data["method"].append("NEURON")

data_df = pd.DataFrame(data)

cols = ["5.7 $\mu m$", "8.7 $\mu m$", "14.0 $\mu m$"]
rows = ["1 kHz", "2 kHz", "5 kHz", "10 kHz"]

fig, axes = plt.subplots(4, 3, dpi=200, figsize=(5, 5), sharex="col", sharey=True)

for i, d_selec in enumerate(all_diams):
    for j, f_selec in enumerate(frequencies):
        data_selection = data_df[data_df["diameter"] == d_selec]
        data_selection = data_selection[data_selection["frequency"] == f_selec]
        sns.lineplot(
            data=data_selection,
            x="amplitude",
            y="# APs",
            hue="method",
            hue_order=["NEURON", "Surrogate"],
            legend=False,
            palette="colorblind",
            ax=axes[j, i],
            errorbar="sd",
        )
        axes[j, i].set_ylabel("")
        axes[j, i].set_xlabel("")

fig.add_subplot(111, frameon=False)

# hide tick and tick label of the big axis
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
plt.xlabel("Amplitude $(mA)$")
plt.ylabel("# APs", labelpad=5)

for ax, col in zip(axes[0], cols):
    ax.set_title(col, size="medium")

for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, rotation=45, size="medium", labelpad=35)

from matplotlib.lines import Line2D

cmap = sns.color_palette("colorblind")
custom_lines = [
    Line2D([0], [0], color=cmap[0], lw=2),
    Line2D([0], [0], color=cmap[1], lw=2),
]
plt.gca().legend(
    custom_lines,
    ["NEURON", "S-MF"],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncols=2,
)

plt.show()

fig.savefig('khz_stim_example.png', bbox_inches='tight')
