from typing import List
import os

import torch
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from .data import DataLoader
from ..models.callbacks import Recorder
from ..models import Axon


class NaNError(Exception):
    pass


def save(model, optimizer, index, validation_error, train_error, directory):
    os.makedirs(directory, exist_ok=True)
    if validation_error is not None:
        save_path = f"{directory}/{index:05d}-{validation_error:.3f}.pt"
    else:
        save_path = f"{directory}/{index:05d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_error,
            "train_error": train_error,
            "index": index,
        },
        save_path,
    )


def tbptt(
    model: Axon,
    data: DataLoader,
    loss_fn: _Loss,
    optimizer: Optimizer,
    epochs: int,
    truncation_length: int,
    dt: float = 0.005,
    grad_accumulation: bool = False,
    postfix: List[str] = None,
    save_every=32,
    save_dir: str = None,
    val_dset: DataLoader = None,
):
    """Run truncated backpropagation through time.

    Parameters
    ----------
    model : Axon
        Surrogate axon model.
    data : DataLoader
        Training data.
    loss_fn : _Loss
        Loss function.
    optimizer : Optimizer
        Optimizer.
    epochs : int
        # epochs.
    truncation_length : int
        # timesteps over which to backpropagate.
    dt : float, optional
        Timestep, by default 0.005
    grad_accumulation : bool, optional
        Use grad accumulation, by default False
    postfix : List[str], optional
        Parameters to print in progressbar, by default None
    save_every : int, optional
        Save every save_every minibatches, by default 32
    save_dir : str, optional
        Save directory, by default None
    val_dset : DataLoader, optional
        Validation data, by default None
    """

    rec = Recorder()
    directory = save_dir if save_dir is not None else "checkpoints"
    zero = torch.tensor(0.0, device=model.device())
    save_idx = 0

    try:
        for _ in range(epochs):
            with tqdm(data.single_epoch(), total=data.total()) as pbar:
                cum_loss = 0
                n = 0
                for i, (x, y, diams) in enumerate(pbar):
                    model.train()
                    n_trunc_chunks = int((x.size(0) / truncation_length) / 4)
                    for j in range(n_trunc_chunks):
                        loss = calc_loss(
                            model, loss_fn, x, y, diams, dt, rec, truncation_length, j
                        )
                        if not torch.isnan(loss):
                            loss.backward()
                            for p in model.parameters():
                                if p.requires_grad:
                                    p.grad = torch.where(
                                        torch.isnan(p.grad), zero, p.grad
                                    )
                            if not grad_accumulation:
                                optimizer.step()
                            cum_loss += loss.item()
                            n += 1
                        if not grad_accumulation:
                            optimizer.zero_grad()

                    if grad_accumulation:
                        optimizer.step()
                        optimizer.zero_grad()

                    pd = {}
                    if postfix is not None:
                        pd = {p: getattr(model, p).item() for p in postfix}
                    pbar.set_postfix(loss=loss.item(), **pd)

                    check_nan(model)

                    if (i == 0) or (i + 1) % save_every == 0:
                        validation_error = None
                        if val_dset is not None:
                            validation_error = validate(model, rec, val_dset, loss_fn)
                        train_error = cum_loss / n
                        save(
                            model,
                            optimizer,
                            save_idx,
                            validation_error,
                            train_error,
                            directory,
                        )
                        cum_loss = 0
                        n = 0
                        save_idx += 1

    except NaNError:
        print("parameter went to NaN")
        for n, p in model.named_parameters():
            print(n, p)


def check_nan(model: Axon):
    """Detect if any parameters have gone to NaN
    (training error).

    Parameters
    ----------
    model : Axon
        Surrogate axon model.

    Raises
    ------
    NaNError
        A parameter is NaN.
    """
    if any(torch.any(torch.isnan(p)) for p in model.parameters()):
        raise NaNError()


def calc_loss(
    model: Axon,
    loss: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    diams: torch.Tensor,
    dt: float,
    rec: Recorder,
    truncation_length: int,
    j: int,
) -> torch.Tensor:
    """Evaluate forward model and calculate loss.

    Parameters
    ----------
    model : Axon
        Surrogate axon model.
    loss : torch.nn.Module
        Loss function.
    x : torch.Tensor
        Input (Ve).
    y : torch.Tensor
        Ground-truth output.
    diams : torch.Tensor
        Fiber diameters (um).
    dt : float
        Timestep (ms).
    rec : Recorder
        Callback for recording states.
    truncation_length : int
        Timsteps over which to perform backpropagation.
    j : int
        Disjoint chunk index.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """

    rec.reset()
    reinit = j == 0
    ve = x[j * truncation_length : (j + 1) * truncation_length]
    model.run(ve, diams, reinit=reinit, dt=dt, callbacks=[rec])
    y_ = y[:, :, j * truncation_length : (j + 1) * truncation_length + 1, :].permute(
        2, 0, 1, 3
    )
    loss_ = loss(y_, rec.stack())
    return loss_


def validate(model: Axon, rec: Recorder, dset: DataLoader, loss: _Loss) -> float:
    """Calculate validation loss.

    Parameters
    ----------
    model : Axon
        Surrogate axon model.
    rec : Recorder
        Callback recorder.
    dset : DataLoader
        Validation dataset (axonml.training.data.DataLoader)
    loss : _Loss
        Loss function.

    Returns
    -------
    float
        Validation loss.
    """

    with tqdm(dset.single_epoch(), total=dset.total()) as pbar:
        cum_loss = 0
        n = 0
        model.eval()
        with torch.no_grad():
            for _, (x, y, diams) in enumerate(pbar):
                rec.reset()
                model.run(x, diams, dt=0.005, reinit=True, callbacks=[rec])
                l_ = loss(rec.stack()[:-1], y[:, :, :, :].permute(2, 0, 1, 3))
                if not torch.isnan(l_):
                    n += 1
                    cum_loss += l_.item()
                    pbar.set_postfix(loss=cum_loss / n)
        return cum_loss / (n + 1)
