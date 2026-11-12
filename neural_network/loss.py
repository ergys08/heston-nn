"""State-fidelity objective from the SuperEncoder implementation."""

import torch


def superencoder_loss(
    target_state: torch.Tensor,
    predicted_state: torch.Tensor,
) -> torch.Tensor:
    """Return L3 = 1 - |<target|predicted>|**2, averaged over a batch.

    Input tensors must have matching shapes: (2**n,) or (batch, 2**n).
    The original SuperEncoder loss converts to complex128 and normalises
    both states before computing the fidelity. We retain that behaviour,
    while preventing accidental flattening of multiple independent states.
    """
    if target_state.shape != predicted_state.shape:
        raise ValueError(
            f"State dimensions do not match: target {tuple(target_state.shape)}, "
            f"predicted {tuple(predicted_state.shape)}."
        )
    if target_state.ndim not in (1, 2):
        raise ValueError("States must have shape (2**n,) or (batch, 2**n).")

    target = target_state.to(torch.complex128)
    predicted = predicted_state.to(torch.complex128)

    target_norms = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
    predicted_norms = torch.linalg.vector_norm(predicted, dim=-1, keepdim=True)
    if torch.any(target_norms == 0) or torch.any(predicted_norms == 0):
        raise ValueError("Cannot compute fidelity for a zero statevector.")

    target = target / target_norms
    predicted = predicted / predicted_norms
    overlap = torch.sum(torch.conj(target) * predicted, dim=-1)
    fidelity = torch.abs(overlap) ** 2
    return (1.0 - fidelity).mean()
