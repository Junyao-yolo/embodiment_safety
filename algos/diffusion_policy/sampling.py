"""Random-permutation sampling with a checkpointed consumed-batch cursor."""
import torch
from torch.utils.data import Sampler,RandomSampler

class ResumableRandomSampler(Sampler):
    def __init__(self,data,state=None):
        self.data=data
        self.order=None if state is None else state["order"]
        self.cursor=0 if state is None else state["cursor"]
    def __iter__(self):
        if self.order is None or self.cursor==len(self.order):
            self.order=list(RandomSampler(self.data))
            self.cursor=0
        yield from self.order[self.cursor:]
    def __len__(self):
        return len(self.data) if self.order is None else len(self.order)-self.cursor
    def advance(self,count):
        self.cursor+=count
    def state_dict(self):
        return dict(order=self.order,cursor=self.cursor)
