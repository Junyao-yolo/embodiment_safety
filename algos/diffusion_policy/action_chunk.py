"""Consume policy plans while respecting teacher ownership of executed actions."""
class ActionChunk:
    def __init__(self, steps):
        self.steps = steps
        self.pending = []
        self.plan_start = None
        self.offset = 0

    def take(self, index, owner, predict):
        if owner != 'policy':
            self.pending.clear()
        inferred = not self.pending
        if inferred:
            self.pending = list(predict())
            if len(self.pending) != self.steps:
                raise ValueError('Policy returned an incomplete action chunk')
            self.plan_start = index
            self.offset = 0
        action = self.pending.pop(0)
        offset = self.offset
        self.offset += 1
        if owner != 'policy':
            self.pending.clear()
        return action, inferred, self.plan_start, offset
