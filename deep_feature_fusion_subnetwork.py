import torch.nn as nn
import torch
from torch.autograd import Variable
class ConvGRUUnit(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super(ConvGRUUnit, self).__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        input_channels=in_channels + hidden_channels
        self.reset_gate = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
    def forward(self, x, prevstate):
        if prevstate is None:
            state_size = [x.shape[0], self.hidden_channels] + list(x.shape[2:])
            prevstate = Variable(torch.zeros(state_size))
            if torch.cuda.is_available():
                prevstate = prevstate.cuda()
        stacked = torch.cat([x, prevstate], dim=1)
        update = torch.sigmoid(self.update_gate(stacked))
        reset = torch.sigmoid(self.reset_gate(stacked))
        candidate_state = torch.tanh(self.out_gate(torch.cat([x, prevstate * reset], dim=1)))
        new_state = (1 - update) * prevstate + update * candidate_state
        return new_state
class deep_feature_fusion_subnetwork(nn.Module):
    def __init__(self):
        super(deep_feature_fusion_subnetwork, self).__init__()
        self.input_channels =1
        self.num_hidden_layers =2
        hidden_channels = [32,32]
        kernel_sizes =[3,3]
        conv_gru = []
        for i in range(0, self.num_hidden_layers):
            cur_input_dim = hidden_channels[i - 1]
            #print(cur_input_dim)
            gru_unit = ConvGRUUnit(in_channels=cur_input_dim, hidden_channels=hidden_channels[i],
                                   kernel_size=kernel_sizes[i])
            conv_gru .append(gru_unit)
        self. conv_gru  = nn.ModuleList( conv_gru )

        self.final = nn.Conv2d(in_channels=32,
                           out_channels=4,
                           kernel_size=1,
                           padding=1 // 2)
    def forward(self, x, a, h=None):
        if h is None:
            hidden_states = [None] * self.num_hidden_layers
        num_low_res = x.shape[1]
        cur_layer_input = x
        for l in range(self.num_hidden_layers):
            gru_unit = self.gru_units[l]
            h = hidden_states[l]
            out = []
            for t in range(num_low_res):
                h = gru_unit(cur_layer_input[:, t, :, :, :], h)
                out.append(h)
            out = torch.stack(out, dim=1)
            cur_layer_input = out
        fused = torch.sum(cur_layer_input * a, 1) / torch.sum(a, 1)
        x = self.final(fused)
        return x

