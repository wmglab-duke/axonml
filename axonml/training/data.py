import numpy as np
import torch

from axonml.models import Backend as A


class DataLoader:
    def __init__(
        self,
        nodes,
        ve,
        states,
        diameters,
        n_idx=None,
        chunk_size=8,
        sampling=None,
        device=None,
        fp32=False,
    ):
        self.nodes = nodes
        self.ve = ve
        self.states = states
        self.diameters = diameters
        self.n_idx = n_idx
        self.chunk_size = chunk_size
        self.sampling = sampling
        self.idxs = list(range(states[0].shape[0]))
        self.device = device if device is not None else A.device
        self.fp32 = fp32
        self.sample_size = diameters.shape[1]

    def single_epoch(self):
        all_vars = self.states
        if not all_vars:
            raise ValueError("Need some states to train against.")

        idxs = self.idxs
        np.random.shuffle(idxs)
        idxs = idxs[: self.n_idx]

        chunk_size = self.chunk_size

        for i in idxs:
            if self.diameters is not None:
                diams_ = self.diameters[i, ...]

            dynamics = []

            for var in all_vars:
                dynamics.append(var[i, ...])

            ve = self.ve[i, ...]

            n_batches = dynamics[0].shape[0]
            n_axons = dynamics[0].shape[1]

            ve = ve.reshape(n_batches * n_axons, self.nodes, 1, -1)

            # -- downsampling --
            if self.sampling:
                for ii, d in enumerate(dynamics):
                    dynamics[ii] = d[:, :, :, :: self.sampling]

            # -- reshape matrices --
            for ii, d in enumerate(dynamics):
                dynamics[ii] = d.reshape(n_batches * n_axons, self.nodes, 1, -1)

            n_chunks = n_batches // chunk_size

            for j in range(n_chunks):
                if self.diameters is not None:
                    diameters_chunk = diams_[j * chunk_size : (j + 1) * chunk_size]

                # -- get dynamics data --
                dynamics_chunks = []
                for d in dynamics:
                    dynamics_chunks.append(
                        d[j * chunk_size * n_axons : (j + 1) * chunk_size * n_axons]
                    )

                # -- build input arrays --
                array = ve[j * chunk_size * n_axons : (j + 1) * chunk_size * n_axons]

                if self.sampling:
                    array = array[:, :, :, :: self.sampling]

                # -- get all ground truth data together
                dynamics_chunk = np.concatenate(dynamics_chunks, axis=2)

                x = (
                    torch.tensor(array, device=self.device)
                    .permute(3, 0, 2, 1)
                    .to(memory_format=torch.contiguous_format)
                )

                y = torch.tensor(dynamics_chunk, device=self.device).permute(0, 2, 3, 1)

                if self.diameters is not None:
                    diams = (
                        torch.tensor(diameters_chunk, device=self.device)
                        .squeeze()
                        .flatten()
                    )
                    if not self.fp32:
                        diams = diams.double()

                if not self.fp32:
                    x = x.double()
                    y = y.double()

                # inputs: (t, b, c, w)
                if self.diameters is not None:
                    yield x, y, diams
                else:
                    yield x, y

    def total(self) -> int:
        return self.n_idx * self.sample_size // self.chunk_size
