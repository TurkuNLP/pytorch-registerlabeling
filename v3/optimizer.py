import torch


class Sophia(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        k=10,
        estimator="Hutchinson",
        rho=1,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            k=k,
            estimator=estimator,
            rho=rho,
        )
        super(Sophia, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")

                state = self.state[p]

                # state init
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["h"] = torch.zeros_like(p.data)

                m, h = state["m"], state["h"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # update biased first moment estimate
                m.mul_(beta1).add_(1 - beta1, grad)

                # update hessian estimate
                if state["step"] % group["k"] == 1:
                    hessian_estimate = self.hutchinson(p, grad)

                    h.mul_(beta2).add_(1 - beta2, hessian_estimate)

                # update params
                p.data.add_(-group["lr"] * group["weight_decay"], p.data)
                p.data.addcdiv_(
                    -group["lr"], m, h.add(group["eps"]).clamp(max=group["rho"])
                )

        return loss

    def hutchinson(self, p, grad):
        u = torch.randn_like(grad)
        hessian_vector_product = torch.autograd.grad(grad.dot(u), p, retain_graph=True)[
            0
        ]
        return u * hessian_vector_product
