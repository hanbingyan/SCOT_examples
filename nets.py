import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class basis_net(nn.Module):
    def __init__(self, in_size, hid_size, out_size, init, req_grad=True):
        super(basis_net, self).__init__()
        self.lstm1 = nn.LSTM(input_size=in_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hid_size, out_size, batch_first=True)

        if init:
            for p in self.lstm1.parameters():
                nn.init.zeros_(p)

            for p in self.lstm2.parameters():
                nn.init.zeros_(p)

        if not req_grad:
            for p in self.lstm1.parameters():
                nn.init.zeros_(p)
                p.requires_grad = False

            for p in self.lstm2.parameters():
                nn.init.zeros_(p)
                p.requires_grad = False


    def forward(self, x):
        # Output shape batch_size*seq_length*hidden_size
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x







class rtn_gen(nn.Module):
    def __init__(self, in_size, out_size):
        super(rtn_gen, self).__init__()
        self.lstm1 = nn.LSTM(input_size=in_size, hidden_size=16, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(16, out_size, batch_first=True)

        for p in self.lstm1.parameters():
            nn.init.zeros_(p)

        for p in self.lstm2.parameters():
            nn.init.zeros_(p)

    def forward(self, x):
        y, _ = self.lstm1(x)
        y, _ = self.lstm2(y)
        return x + y


class mv_obj(nn.Module):
    def __init__(self, weight_init, mean_init, risk_aversion):
        super(mv_obj, self).__init__()
        self.weight = nn.Parameter(weight_init)
        self.mean = nn.Parameter(mean_init)
        self.risk = torch.tensor(risk_aversion, requires_grad=False, dtype=torch.float32, device=device)

        # for p in self.layers.parameters():
        #     nn.init.uniform_(p, 0.0, 0.1)
        #     p.requires_grad = False

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.prod(1 + x, dim=1)
        x = (x-1-self.mean)**2 - self.risk*(x-1)
        return x

# if __name__ == '__main__':
#
#     x = torch.ones(1, 1, 2, device=device)
#     x[:, :, 0] = 0.1
#     x[:, :, 1] = 0.2
#     f = mv_obj(weight_init=torch.tensor([0.6, 0.4]), mean_init=torch.ones(1), risk_aversion=0.0).to(device)
#     print(f(x))
