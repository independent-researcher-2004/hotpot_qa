import torch 
from torch.autograd import Variable 
from torch import nn 

# Original implementation:
# https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
# Why didn't they write the shout out for the original code ? 


class LockedDropout(nn.Module): 
    """ LockedDropout applies the same dropout mask to every time step.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        dropout (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, dropout): 
        super().__init__()
        self.dropout = dropout
        
    def forward(self, x): 
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        dropout = self.dropout 
        if not self.training: 
            return x 
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout)
        mask = mask.expand_as(x)
        return x * mask

if __name__ == "__main__":
    print()