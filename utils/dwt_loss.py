import torch
from pytorch_wavelets import DWTForward


def dwt_transform(latent):
    """
    Decompose input latent using discrete wavelet transform,
    and return the wavelets stacked along the channel dimension
    
    Args: latent (torch.Tensor): latent image with shape (B, C, H, W)
    
    Returns: torch.Tensor: wavelet decomposed latent with shape (B, C*4, H, W)
    """
    assert latent.dim() == 4
    dwt = DWTForward(J=1, mode='zero', wave='haar').to(device=latent.device, dtype=latent.dtype)
    
    latent_xll, latent_xh = dwt(latent)
    latent_xlh, latent_xhl, latent_xhh = torch.unbind(latent_xh[0], dim=2) # split along the extra dim
    
    return torch.cat([latent_xll, latent_xlh, latent_xhl, latent_xhh], dim=1) # concat along channel dim


def dwt_loss(target, pred, reduction="mean"):
    """
    Calculate MSE loss in wavelet domain
    
    Args:
        target (torch.Tensor): clean latent image with shape (B, C, H, W)
        pred (torch.Tensor): clean latent image with shape (B, C, H, W)
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    
    Returns: torch.Tensor: MSE loss in wavelet domain
    """
    assert target.shape == pred.shape
    
    dwt_target = dwt_transform(target)
    dwt_pred = dwt_transform(pred)
    
    return torch.nn.functional.mse_loss(dwt_target, dwt_pred, reduction=reduction)