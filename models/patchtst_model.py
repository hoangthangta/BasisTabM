
from types import SimpleNamespace
from .patchtst_official.PatchTST import Model


class PatchTSTModel(Model):
    def __init__(self, input_dim=1, seq_len=144, horizon=1, patch_len=16, stride=8,
                 d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1,
                 fc_dropout=0.05, head_dropout=0.0, revin=True, affine=False,
                 subtract_last=False, padding_patch="end", individual=False,
                 decomposition=False, kernel_size=25):
        cfg = SimpleNamespace(
            enc_in=input_dim, seq_len=seq_len, pred_len=horizon,
            e_layers=e_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff,
            dropout=dropout, fc_dropout=fc_dropout, head_dropout=head_dropout,
            individual=individual, patch_len=patch_len, stride=stride,
            padding_patch=padding_patch, revin=revin, affine=affine,
            subtract_last=subtract_last, decomposition=decomposition,
            kernel_size=kernel_size
        )
        super().__init__(cfg)

    def forward(self, x):
        y = super().forward(x)  # [B, horizon, input_dim]
        return y.squeeze(-1) if y.shape[-1] == 1 else y
