import math

import torch


def zeropower_via_newtonschulz5(x: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("Muon orthogonalization expects a 2D tensor")
    if x.numel() == 0:
        return x

    original_dtype = x.dtype
    x = x.float()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x

    if transposed:
        x = x.T
    return x.to(original_dtype)


class Muon(torch.optim.Optimizer):
    """Momentum optimizer for 2D hidden weights using Newton-Schulz orthogonalization."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon only supports 2D parameters")

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))

                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr * scale)
        return loss


class CombinedOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)


def uses_muon(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim != 2:
        return False
    if name == "tok_embeddings.weight" or name == "lm_head.weight":
        return False
    if ".ple." in name or name.startswith("ple."):
        return False
    return True


def build_optimizer(model: torch.nn.Module, train_config: dict) -> torch.optim.Optimizer | CombinedOptimizer:
    name = train_config.get("optimizer", "adamw").lower()
    lr = train_config.get("lr", 3e-4)
    weight_decay = train_config.get("weight_decay", 0.0)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name != "muon":
        raise ValueError(f"unknown optimizer: {name}")

    muon_params = []
    adamw_params = []
    for param_name, param in model.named_parameters():
        if uses_muon(param_name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    optimizers: list[torch.optim.Optimizer] = []
    if muon_params:
        optimizers.append(
            Muon(
                muon_params,
                lr=train_config.get("muon_lr", 0.02),
                momentum=train_config.get("muon_momentum", 0.95),
                weight_decay=weight_decay,
                ns_steps=train_config.get("muon_ns_steps", 5),
                nesterov=train_config.get("muon_nesterov", True),
            )
        )
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                adamw_params,
                lr=train_config.get("adamw_lr", lr),
                weight_decay=weight_decay,
            )
        )
    return CombinedOptimizer(optimizers)
