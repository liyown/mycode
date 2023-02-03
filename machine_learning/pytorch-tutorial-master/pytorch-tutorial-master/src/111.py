import copy

from torch import nn


def fn(b):
    print(b)
    print(b.__class__.__name__)


class aaa(nn.Module):
    def __init__(self):
        super(aaa, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),

            nn.Conv2d(32, 32, 5, 1, 2),

            nn.Conv2d(32, 64, 5, 1, 2),
            nn.Linear(64, 10)
        )




a = aaa()
b = copy.deepcopy(a)
# a.apply(fn)

print()
