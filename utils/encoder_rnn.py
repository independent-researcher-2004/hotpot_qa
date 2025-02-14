import torch 
from torch import nn 
from .locked_dropout import LockedDropout
from torch.nn.utils import rnn 

class EncoderRNN(nn.Module): 
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last): 
        super().__init__()
        self.rnns = []
        for i in range(nlayers): 
            if i == 0: 
                input_size_ = input_size 
                output_size_ = num_units 
            else: 
                input_size_ = num_units if not bidir else num_units * 2 
                output_size_ = num_units 
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat 
        self.return_last = return_last 
        
        # self.reset_parameters()
        
    def reset_parameters(self): 
        for rnn in self.rnns: 
            for name, p in rnn.named_parameters(): 
                if 'weight' in name: 
                    p.data.normal_(std=0.1)
                else: 
                    p.data.zero_()
    
    def get_init(self, bsz, i): 
        return self.init_hidden[i].expand(-1, bsz, -1).contigous()
    
    def forward(self, input, input_lengths=None): 
        bsz, slen = input.size(0), input.size(1)
        output = input 
        outputs = []
        if input_lengths is not None: 
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers): 
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            
            if input_lengths is not None: 
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
            
            if input_lengths is not None: 
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel 
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim = 1)
            if self.return_last: 
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1)) 
            else: 
                outputs.append(output)
        if self.concat: 
            return torch.cat(outputs, dim=2)
        return outputs[-1]