"""Mutation Preference Optimization: calibrated signs and program preferences."""

import torch
import torch.nn.functional as F

from .common import DEVICE


class SelectorObjective:
    """Strongly convex task classification and program-ranking objective."""

    def __init__(self, x, eps, ids, records, cfg):
        self.x = x[ids]
        labels = (eps[ids] > 0).double()
        weights = torch.where(labels.bool(), 0.5 / labels.sum(), 0.5 / (1 - labels).sum())
        first, second = records["first"], records["second"]
        difference = eps[first] - eps[second]
        valid = difference != 0
        delta = x[first[valid]] - x[second[valid]]
        self.a = torch.cat([self.x, delta])
        temperature = cfg.get("rank_temperature", 0.0)
        preference = (
            torch.sigmoid(difference[valid] / temperature)
            if temperature
            else (difference[valid] > 0).double()
        )
        self.p = torch.cat([labels, preference])
        self.weights = torch.cat(
            [
                cfg.get("classification", 1.0) * weights,
                x.new_full((len(delta),), cfg["preference"]) / len(delta),
            ]
        )
        self.mu = cfg["ridge"]
        self.dimension = x.shape[1]
        self.calibration = cfg.get("calibration", 0.0)
        self.calibration_target = eps[ids] / (temperature if temperature else 1.0)
        self.delta = 1.0

    def loss(self, theta):
        logits = self.a @ theta
        binary = self.p * F.softplus(-logits, threshold=40) + (1 - self.p) * F.softplus(
            logits, threshold=40
        )
        residual = (self.x @ theta - self.calibration_target) / self.delta
        robust = self.calibration * self.delta**2 * (torch.sqrt(1 + residual.square()) - 1).mean()
        return self.mu * theta.square().sum() / 2 + self.weights @ binary + robust

    def gradient(self, theta):
        residual = (self.x @ theta - self.calibration_target) / self.delta
        robust = (
            self.calibration
            * self.delta
            / len(self.x)
            * self.x.T
            @ (residual / torch.sqrt(1 + residual.square()))
        )
        return (
            self.mu * theta
            + self.a.T @ (self.weights * (torch.sigmoid(self.a @ theta) - self.p))
            + robust
        )

    def hessian(self, theta):
        probability = torch.sigmoid(self.a @ theta)
        residual = (self.x @ theta - self.calibration_target) / self.delta
        robust = (
            self.calibration
            / len(self.x)
            * self.x.T
            @ ((1 + residual.square()).pow(-1.5)[:, None] * self.x)
        )
        return (
            self.mu * torch.eye(self.dimension, device=DEVICE, dtype=torch.float64)
            + self.a.T @ ((self.weights * probability * (1 - probability))[:, None] * self.a)
            + robust
        )

    def fit(self, start=None, iterations=180):
        theta = self.a.new_zeros(self.dimension) if start is None else start.detach().clone()
        theta.requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [theta],
            lr=1.0,
            max_iter=iterations,
            history_size=30,
            tolerance_grad=1e-8,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = self.loss(theta)
            loss.backward()
            return loss

        optimizer.step(closure)
        return theta.detach(), {
            "gradient": self.gradient(theta.detach()).norm().item(),
            "loss": self.loss(theta.detach()).item(),
            "iterations": optimizer.state[theta]["n_iter"],
        }
