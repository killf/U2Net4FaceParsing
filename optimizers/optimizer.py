import torch
import logging

logger = logging.getLogger()


class SGDOptimizer:
    def __init__(self, model, lr0, momentum, weight_decay, warmup_steps, warmup_start_lr, max_iter, power, *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0

        self.optim = torch.optim.SGD(model.parameters(), lr=lr0, momentum=momentum, weight_decay=weight_decay)
        self.warmup_factor = (self.lr0 / self.warmup_start_lr) ** (1. / self.warmup_steps)

    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr * (self.warmup_factor ** self.it)
        else:
            factor = (1 - (self.it - self.warmup_steps) / (self.max_iter - self.warmup_steps)) ** self.power
            lr = self.lr0 * factor
        return lr

    def step(self):
        self.adjust_lr(self.get_lr())
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps + 2:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def adjust_lr(self, lr):
        self.lr = lr
        for pg in self.optim.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = self.lr * 10
            else:
                pg['lr'] = self.lr
        if self.optim.defaults.get('lr_mul', False):
            self.optim.defaults['lr'] = self.lr * 10
        else:
            self.optim.defaults['lr'] = self.lr

    def zero_grad(self):
        self.optim.zero_grad()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        return self.optim.load_state_dict(state_dict)
