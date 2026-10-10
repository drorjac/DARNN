"""Common block predictors and a PyTorch adaptation of Qin et al. (2017)."""
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, kind, history, width):
        super().__init__()
        self.kind, self.history = kind, history
        if kind == 'mlp':
            self.net = nn.Sequential(nn.Linear(history, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        elif kind in {'lstm', 'gru'}:
            self.net = getattr(nn, kind.upper())(1, width, batch_first=True)
        elif kind == 'darnn':
            self.net = DARNN(history, width)
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == 'mlp':
            return self.net(F.adaptive_avg_pool1d(x.transpose(1, 2), self.history).squeeze(1))
        if self.kind == 'darnn':
            return self.net(x)
        return self.net(x)[0][:, -1]


class DARNN(nn.Module):
    """Input attention and temporal decoder attention over an observed context.

    For a single feature input attention is identically one. Feature trajectories
    are pooled to the configured history length for expanding contexts; fixed
    rolling contexts retain the original attention formulation.
    """
    def __init__(self, history, width):
        super().__init__()
        self.history, self.width = history, width
        self.feature = nn.Linear(history, width)
        self.input_query = nn.Linear(2*width, width)
        self.input_score = nn.Linear(width, 1)
        self.encoder = nn.LSTMCell(1, width)
        self.temporal_keys = nn.Linear(width, width)
        self.temporal_query = nn.Linear(2*width, width)
        self.temporal_score = nn.Linear(width, 1)
        self.decoder_input = nn.Linear(width+1, 1)
        self.decoder = nn.LSTMCell(1, width)
        self.fusion = nn.Linear(2*width, width)

    def forward(self, x):
        h = x.new_zeros(len(x), self.width)
        c = torch.zeros_like(h)
        features = F.adaptive_avg_pool1d(x.transpose(1, 2), self.history)
        keys = self.feature(features)
        encoded = []
        for value in x.unbind(1):
            score = self.input_score(torch.tanh(keys+self.input_query(torch.cat([h, c], -1))[:, None]))
            attention = score.softmax(1).squeeze(-1)
            h, c = self.encoder(attention*value, (h, c))
            encoded.append(h)
        encoded = torch.stack(encoded, 1)
        keys = self.temporal_keys(encoded)
        d, cell = torch.zeros_like(h), torch.zeros_like(h)
        for value in x.unbind(1):
            query = self.temporal_query(torch.cat([d, cell], -1))[:, None]
            weights = self.temporal_score(torch.tanh(keys+query)).softmax(1)
            context = (weights*encoded).sum(1)
            d, cell = self.decoder(self.decoder_input(torch.cat([value, context], -1)), (d, cell))
        return self.fusion(torch.cat([d, context], -1))


class BlockPredictor(nn.Module):
    def __init__(self, kind, history, window, width=32, separate=False):
        super().__init__()
        self.encoder = Encoder(kind, history, width)
        self.separate = separate
        if separate:
            self.pred_encoder = Encoder(kind, history, width)
        self.head = nn.Linear(width*(2 if separate else 1), window)

    def forward(self, observed, predicted=None):
        state = self.encoder(observed)
        if self.separate:
            previous = torch.zeros_like(state) if predicted is None else self.pred_encoder(predicted)
            state = torch.cat([state, previous], -1)
        return self.head(state).unsqueeze(-1)+observed[:, -1:]
