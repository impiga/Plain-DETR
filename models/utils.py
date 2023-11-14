# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Sample2D(nn.Module):
    def __init__(self, stride, start):
        super().__init__()
        self.stride = stride
        self.start = start

    def forward(self, x):
        """
        x: N C H W
        """
        _, _, h, w = x.shape
        return x[:, :, self.start: h: self.stride, self.start: w: self.stride]

    def extra_repr(self) -> str:
        return f'stride={self.stride}, start={self.start}'


class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ln = norm_layer(normalized_shape) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        x: N C H W
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


def load_swinv2_checkpoint(model, filename, map_location="cpu", strict=False):
    """Load swin checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if filename.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            filename, map_location=map_location, check_hash=True
        )
    else:
        checkpoint = torch.load(filename, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # strip prefix for mim ckpt
    if any([k.startswith("encoder.") for k in state_dict.keys()]):
        print("Remove encoder. prefix")
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }

    # rename rpe to cpb (naming inconsistency of sup & mim ckpt)
    if any(["rpe_mlp" in k for k in state_dict.keys()]):
        print("Replace rpe_mlp with cpb_mlp")
        state_dict = {k.replace("rpe_mlp", "cpb_mlp"): v for k, v in state_dict.items()}

    # remove relative_coords_table & relative_position_index in state_dict as they would be re-init
    if any(["relative_coords_table" in k or "relative_position_index" in k
            for k in state_dict.keys()]):
        print("Remove relative_coords_table & relative_position_index (they would be re-init)")
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if "relative_coords_table" not in k and "relative_position_index" not in k
        }

    # reshape absolute position embedding
    if state_dict.get("absolute_pos_embed") is not None:
        absolute_pos_embed = state_dict["absolute_pos_embed"]
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            print("Warning: Error in loading absolute_pos_embed, pass")
        else:
            state_dict["absolute_pos_embed"] = absolute_pos_embed.view(
                N2, H, W, C2
            ).permute(0, 3, 1, 2)

    # load state_dict
    msg = model.load_state_dict(state_dict, strict=strict)
    print(msg)
    return checkpoint
