"""components of SwinTransformer."""
import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import numpy as np


class CyclicShift(nn.Module):
    """Cyclic Shift from the paper."""

    def __init__(self, displacement: int) -> None:
        """init method.

        Args:
            displacement (int): _description_
        """
        super().__init__()
        self.diplacement = displacement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return torch.roll(x, shifts=(self.diplacement, self.diplacement), dims=(1, 2))


class Residual(nn.Module):
    """resiual op."""

    def __init__(self, fn: nn.Module):
        """residual block.

        Args:
            fn (nn.Module): block layer.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_
            **kwargs :_description_

        Returns:
            torch.Tensor: _description_
        """
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """PreNorm Module."""

    def __init__(self, dim: int, fn: nn.Module):
        """init method.

        Args:
            dim (int): _description_
            fn (nn.Module): _description_
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_
            **kwargs :_description_

        Returns:
            torch.Tensor: _description_
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """FeedForward Module."""

    def __init__(self, dim: int, hidden_dim: int):
        """init methpod.

        Args:
            dim (int): _description_
            hidden_dim (int): _description_
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.net(x)


def create_mask(
    window_size: int, displacement: int, upper_lower: bool, left_right: bool
) -> torch.Tensor:
    """create mask for attenton.

    Args:
        window_size (int): _description_
        displacement (int): _description_
        upper_lower (bool): _description_
        left_right (bool): _description_

    Returns:
        torch.Tensor: _description_
    """
    mask = torch.zeros(window_size**2, window_size**2)

    if upper_lower:
        mask[-displacement * window_size :, : -displacement * window_size] = float(
            "-inf"
        )
        mask[: -displacement * window_size, -displacement * window_size :] = float(
            "-inf"
        )

    if left_right:
        mask = rearrange(
            mask, "(h1 w1) (h2 w2) -> h1 w1 h2 w2", h1=window_size, h2=window_size
        )
        mask[:, -displacement:, :, :-displacement] = float("-inf")
        mask[:, :-displacement, :, -displacement:] = float("-inf")
        mask = rearrange(mask, "h1 w1 h2 w2 -> (h1 w1) (h2 w2)")

    return mask


def get_relative_distances(window_size: int) -> torch.Tensor:
    """get relative distance.

    Args:
        window_size (int): _description_

    Returns:
        torch.Tensor: _description_
    """
    indices = torch.tensor(
        np.array([[x, y] for x in range(window_size) for y in range(window_size)])
    )
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    """Window Attention."""

    def __init__(
        self, dim: int, heads: int, head_dim: int, shifted: bool, window_size: int
    ) -> None:
        """Window Attention.

        Args:
            dim (int): _description_
            heads (int): _description_
            head_dim (int): _description_
            shifted (bool): _description_
            window_size (int): _description_
        """
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** (-0.5)
        self.window_size = window_size
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(displacement=-displacement)
            self.cyclic_back_shift = CyclicShift(displacement=displacement)

            self.upper_lower_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=True,
                    left_right=False,
                ),
                requires_grad=False,
            )

            self.left_right_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=False,
                    left_right=True,
                ),
                requires_grad=False,
            )

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # Relative position bias from the paper.
        self.relative_indices = get_relative_distances(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(
            torch.randn(2 * window_size - 1, 2 * window_size - 1)
        )
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d",
                h=h,
                w_h=self.window_size,
                w_w=self.window_size,
            ),
            qkv,
        )

        dots = einsum("b h w i d, b h w j d -> b h w i j", q, k) * self.scale

        dots += self.pos_embedding[
            self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]
        ]

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1 :: nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum("b h w i j, b h w j d -> b h w i d", attn, v)
        out = rearrange(
            out,
            "b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)",
            h=h,
            w_h=self.window_size,
            w_w=self.window_size,
            nw_h=nw_h,
            nw_w=nw_w,
        )
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    """Swin block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        mlp_dim: int,
        shifted: bool,
        window_size: int,
    ) -> None:
        """init method.

        Args:
            dim (int): _description_
            heads (int): _description_
            head_dim (int): _description_
            mlp_dim (int): _description_
            shifted (bool): _description_
            window_size (int): _description_
        """
        super().__init__()
        self.atttention_block = Residual(
            PreNorm(
                dim=dim,
                fn=WindowAttention(
                    dim=dim,
                    heads=heads,
                    head_dim=head_dim,
                    shifted=shifted,
                    window_size=window_size,
                ),
            )
        )
        self.mlp_block = Residual(
            PreNorm(dim=dim, fn=FeedForward(dim=dim, hidden_dim=mlp_dim))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.atttention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    """PatchMerging class."""

    def __init__(
        self, in_channels: int, out_channels: int, downscaling_factor: int
    ) -> None:
        """PatchMerging.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            downscaling_factor (int): _description_
        """
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(
            kernel_size=downscaling_factor, stride=downscaling_factor, padding=0
        )
        self.layer_norm = nn.LayerNorm(in_channels * downscaling_factor**2)
        self.linear = nn.Linear(in_channels * downscaling_factor**2, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        b, c, h, w = x.shape
        out_h, out_w = h // self.downscaling_factor, h // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, out_h, out_w).permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    """strage Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        layers: int,
        downscaling_factor: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
    ) -> None:
        """init method.

        Args:
            in_channels (int): _description_
            hidden_dim (int): _description_
            layers (int): _description_
            downscaling_factor (int): _description_
            num_heads (int): _description_
            head_dim (int): _description_
            window_size (int): _description_
        """
        super().__init__()
        self.patch_merging = PatchMerging(
            in_channels=in_channels,
            out_channels=hidden_dim,
            downscaling_factor=downscaling_factor,
        )
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(
                nn.ModuleList(
                    [
                        SwinBlock(
                            dim=hidden_dim,
                            heads=num_heads,
                            head_dim=head_dim,
                            window_size=window_size,
                            shifted=False,
                            mlp_dim=hidden_dim * 4,
                        ),
                        SwinBlock(
                            dim=hidden_dim,
                            heads=num_heads,
                            head_dim=head_dim,
                            mlp_dim=hidden_dim * 4,
                            shifted=True,
                            window_size=window_size,
                        ),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.patch_merging(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)

        x = x.permute(0, 3, 1, 2)
        return x
