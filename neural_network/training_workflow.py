"""Train a SuperEncoder on real state amplitudes; evaluate future Heston states.

All synthetic augmentation is performed elsewhere in state space. The Heston
CSV is used ONLY to obtain the original dated anchor/test quantum states.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from circuits import run_brickwork_circuit, run_chain_circuit, run_ring_circuit
from heston.distributions import HestonParameters, heston_pdf, discretize_pdf
from .loss import superencoder_loss
from .mlp import SuperEncoderMLP

CIRCUITS = {
    "brickwork": run_brickwork_circuit,
    "chain": run_chain_circuit,
    "ring": run_ring_circuit,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_heston_states(path: str | Path, *, num_qubits: int = 8) -> tuple[list[str], np.ndarray]:
    """Load dated calibrations as 256-dimensional sqrt-probability amplitudes."""
    dates, states = [], []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "v0", "kappa", "theta", "sigma", "rho"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing Heston CSV columns: {required - set(reader.fieldnames or [])}")
        for row in reader:
            day = date.fromisoformat(row["date"])
            params = HestonParameters(
                v0=float(row["v0"]), kappa=float(row["kappa"]),
                theta=float(row["theta"]), xi=float(row["sigma"]),
                rho=float(row["rho"]),
            )
            try:
                x, pdf, _ = heston_pdf(params)
                _, _, probs, _ = discretize_pdf(x, pdf, n_qubits=num_qubits, cutoff=2.5)
            except (ArithmeticError, ValueError) as error:
                raise ValueError(f"Could not construct Heston state for {day}: {error}") from error
            dates.append(day.isoformat())
            states.append(np.sqrt(probs).astype(np.float64))
    if not dates or any(a >= b for a, b in zip(dates, dates[1:])):
        raise ValueError("Calibration rows must have strictly increasing dates.")
    return dates, np.stack(states)


def train_network(
    training_states: np.ndarray,
    *,
    num_qubits: int = 8,
    num_layers: int = 18,
    topology: str = "brickwork",
    rotation: str = "yz",
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device: str = "cpu",
    progress_path: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> tuple[SuperEncoderMLP, list[dict[str, float]]]:
    """Vectorised SuperEncoder fitting; never sees held-out future states."""
    if topology not in CIRCUITS:
        raise ValueError("Unknown topology.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    seed_everything(seed)
    target = torch.tensor(np.asarray(training_states), dtype=torch.float32)
    if target.ndim != 2 or target.shape[1] != (1 << num_qubits):
        raise ValueError("training_states must be of shape (N, 2**num_qubits).")
    if not torch.isfinite(target).all() or (torch.linalg.vector_norm(target, dim=1) == 0).any():
        raise ValueError("Invalid training states.")
    target = target / torch.linalg.vector_norm(target, dim=1, keepdim=True)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(target), batch_size=batch_size, shuffle=True,
        generator=loader_generator,
    )
    net = SuperEncoderMLP(num_qubits, num_layers, topology, rotation).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    circuit = CIRCUITS[topology]
    history = []
    start_epoch = 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=True)
        net.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer_state"])
        history = list(state["history"])
        start_epoch = int(state["epoch"])
        if start_epoch > epochs:
            raise ValueError("Resume checkpoint is beyond the requested epoch count.")
        if "loader_generator_state" in state:
            loader_generator.set_state(state["loader_generator_state"])
        else:
            # Backward-compatible resume for checkpoints before generator saving.
            for _ in range(start_epoch):
                for _ in loader:
                    pass
        print(f"Resuming from completed epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch + 1, epochs + 1):
        net.train()
        sum_loss, seen = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            angles = net(batch)  # B x P; target amplitudes -> circuit angles
            prepared = circuit(num_qubits, num_layers, angles, rotation=rotation)
            loss = superencoder_loss(batch, prepared)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * len(batch)
            seen += len(batch)
        avg_loss = sum_loss / seen
        history.append({"epoch": epoch, "train_loss": avg_loss, "train_fidelity": 1 - avg_loss})
        print(f"Epoch {epoch:3d}/{epochs} | train loss {avg_loss:.6f} | "
              f"train fidelity {1 - avg_loss:.6f}", flush=True)
        if progress_path is not None:
            torch.save({"state_dict": net.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch, "history": history,
                        "loader_generator_state": loader_generator.get_state()}, progress_path)
    return net, history


@torch.no_grad()
def evaluate_network(
    model: SuperEncoderMLP,
    target_states: np.ndarray,
    *,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Evaluate fidelity on unseen states without adapting model parameters."""
    targets = torch.tensor(np.asarray(target_states), dtype=torch.float32)
    targets = targets / torch.linalg.vector_norm(targets, dim=1, keepdim=True)
    model.eval()
    circuit = CIRCUITS[model.topology]
    results = []
    for chunk in targets.split(batch_size):
        chunk = chunk.to(device)
        angles = model(chunk)
        prepared = circuit(model.num_qubits, model.num_layers, angles, rotation=model.rotation)
        overlap = torch.sum(torch.conj(chunk.to(torch.complex128)) * prepared, dim=-1)
        fidelity = torch.abs(overlap).square()
        results.extend(fidelity.cpu().numpy().tolist())
    return np.asarray(results, dtype=np.float64)


def save_checkpoint(model: SuperEncoderMLP, path: str | Path, *, cutoff_date: str, seed: int) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "num_qubits": model.num_qubits,
        "num_layers": model.num_layers,
        "topology": model.topology,
        "rotation": model.rotation,
        "training_cutoff_date": cutoff_date,
        "seed": seed,
    }, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> SuperEncoderMLP:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = SuperEncoderMLP(
        ckpt["num_qubits"], ckpt["num_layers"], ckpt["topology"], ckpt["rotation"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def nearest_fs_distances(states: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Nearest training anchor distance, using real nonnegative amplitudes."""
    states = np.asarray(states, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    dots = states @ anchors.T
    norms = np.linalg.norm(states, axis=1)[:, None] * np.linalg.norm(anchors, axis=1)[None, :]
    similarities = np.clip(np.abs(dots / norms), 0, 1)
    return np.arccos(np.max(similarities, axis=1))
