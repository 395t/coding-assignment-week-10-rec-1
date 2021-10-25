from collections import Counter
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler


class WarmUpMultiStepLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer,
                 milestones: List[int],
                 gamma: float = 0.1,
                 warm_up_factor: float = 0.1,
                 warm_up_iters: int = 500,
                 last_epoch: int = -1,
                 verbose: bool = False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.factor = warm_up_factor
        self.warm_up_iters = warm_up_iters
        self.lrs = [group['lr'] for group in optimizer.param_groups]
        super(WarmUpMultiStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch in self.milestones:
            self.lrs = [lr * self.gamma ** self.milestones[self.last_epoch] for lr in self.lrs]
        if self.last_epoch < self.warm_up_iters:
            alpha = self.last_epoch / self.warm_up_iters
            factor = (1 - self.factor) * alpha + self.factor
        else:
            factor = 1
        lrs = [lr * factor for lr in self.lrs]
        return lrs


if __name__ == '__main__':
    last_epoch = 2
    for iter in range(1, 1000):
        factor = 0.1
        alpha = iter / 1000
        factor = (1 - factor) * alpha + factor
        print(f'factor:{factor}')
