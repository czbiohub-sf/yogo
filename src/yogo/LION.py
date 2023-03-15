""" :lion: :lion: :lion:
"""

import torch


from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr=4e-5,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay=1e-2,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # for Optimizer
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
        )

        super(Lion, self).__init__(params)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )

        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    @torch.no_grad()
    def step(self, closure=None):
        pass
