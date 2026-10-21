"""Selective Epistasis Forgetting with protected curvature correction."""

import torch

from .common import DEVICE


def selective_forgetting(x, req, objective, original):
    """Protected Newton displacement followed by retained-objective convergence."""
    gradient = objective.gradient(original)
    hessian = objective.hessian(original)
    newton = torch.linalg.solve(hessian, -gradient)
    protected_ids = sorted(set(req["retained"]) & set(req["adjacent"]))
    protected = x[protected_ids]
    target = x[req["forgotten"]]
    metric = 0.01 * torch.eye(
        x.shape[1], device=DEVICE, dtype=torch.float64
    )
    if len(protected):
        metric = metric + protected.T @ protected / len(protected)
    inverse = torch.linalg.solve(metric, target.T)
    constrained = inverse @ (
        torch.linalg.pinv(target @ inverse, rtol=1e-9, hermitian=True) @ (target @ newton)
    )
    displacement = 0.75 * newton + 0.25 * constrained
    theta, details = objective.fit(original + displacement, iterations=600)
    details.update(
        {
            "protected_count": len(protected_ids),
            "omega": 0.25,
            "constraint_residual": (target @ (constrained - newton)).norm().item(),
            "initial_retained_gradient": gradient.norm().item(),
        }
    )
    return theta, details
