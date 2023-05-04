import torch
import torch.nn as nn


class HMRLayer(nn.Module):
    def __init__(self, feat_dim, mid_dim, specs_dict):
        super().__init__()

        self.feat_dim = feat_dim
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.specs_dict = specs_dict

        vector_dim = sum(list(zip(*specs_dict.items()))[1])
        hmr_dim = feat_dim + vector_dim

        # construct refine
        self.refine = nn.Sequential(
            nn.Linear(hmr_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(),
        )

        # construct decoders
        decoders = {}
        for key, vec_size in specs_dict.items():
            decoders[key] = nn.Linear(mid_dim, vec_size)
        self.decoders = nn.ModuleDict(decoders)

        self.init_weights()

    def init_weights(self):
        for key, decoder in self.decoders.items():
            nn.init.xavier_uniform_(decoder.weight, gain=0.01)
            self.decoders[key] = decoder

    def forward(self, feat, init_vector_dict, n_iter):
        pred_vector_dict = init_vector_dict
        for i in range(n_iter):
            vectors = list(zip(*pred_vector_dict.items()))[1]
            xc = torch.cat([feat] + list(vectors), dim=1)
            xc = self.refine(xc)
            for key, decoder in self.decoders.items():
                pred_vector_dict.overwrite(key, decoder(xc) + pred_vector_dict[key])

        pred_vector_dict.has_invalid()
        return pred_vector_dict
