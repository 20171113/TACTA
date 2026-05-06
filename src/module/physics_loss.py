import torch

def compute_force(
    Fx0, Fy0, VB,
    VB_star,feed, tool_dia, tool_radi, axial_depth, radial_depth, helix_angle,
    kr, fz, r, HV, v1, v2, E1, E2, R1, alpha):
    
    """
    Compute physics-based cutting forces using Waldorf's wear-force model.
    Equation (4) ~ (11)
    """
    
    device = Fx0.device
    
    alpha = alpha.to(device)
    helix_angle = helix_angle.to(device)

    phi = torch.atan2(Fy0, Fx0)
    Fc = Fx0 * torch.cos(phi) - Fy0 * torch.sin(phi)
    Ft = Fx0 * torch.sin(phi) + Fy0 * torch.cos(phi)

    bun = Fc * torch.sin(alpha) + Ft * torch.cos(alpha)
    mo = Fc * torch.cos(alpha) - Ft * torch.sin(alpha)
    mu = bun / mo
    
    beta = torch.atan(mu)
    shearangle = torch.pi / 4 - beta / 2 + alpha / 2

    F_shear = torch.cos(shearangle) * torch.cos(beta - alpha) * Fc
    l_shear = fz / torch.sin(shearangle)
    
    arc_arg = torch.tensor(1 - radial_depth / tool_radi, device=device)
    arc_arg = torch.clamp(arc_arg, min=-1.0, max=1.0)
    lc = tool_dia / 2 * torch.arccos(arc_arg)
    
    w = lc / torch.cos(helix_angle)
    tau_shear = F_shear / (l_shear * w)
    k = tau_shear

    Fx_total = torch.zeros_like(Fx0)
    Fy_total = torch.zeros_like(Fy0)

    mask = VB <= VB_star
    mask_inv = ~mask

    # VB <= VB_star (Elastic only)
    if mask.any():
        VB1 = VB[mask]
        mu1 = mu[mask]
        Fc1 = Fc[mask]
        Ft1 = Ft[mask]
        shearangle1 = shearangle[mask]
        k1 = k[mask]
        phi1 = phi[mask]
    
        ita_p = 0.5 * torch.arccos(torch.tensor(0.99, device=device))
        # rho = 0 (prow angle, negligible)
        gamma = ita_p + shearangle1 - torch.arcsin(
            torch.sqrt(torch.tensor(2.0, device=device)) * torch.sin(ita_p) * torch.sin(torch.tensor(0.0, device=device))
        )

        tau_0 = k1 * torch.cos(2 * gamma - 2 * shearangle1)
        sigma_0 = k1 * (1 + torch.pi / 2 - 2 * shearangle1 + 2 * gamma + torch.sin(2 * gamma - 2 * shearangle1))
        kk = 1 - torch.sqrt(tau_0 / sigma_0)
        
        # Integration results
        F_tw_1 = sigma_0 * VB1 / 3 * axial_depth
        F_cw_1 = (tau_0 * VB1 * kk + mu1 * sigma_0 * VB1 * (1 - kk)**3 / 3) * axial_depth
        
        F_c_total = Fc1 + F_cw_1
        F_t_total = Ft1 + F_tw_1
    
        Fx_total[mask] = F_c_total * torch.cos(phi1) + F_t_total * torch.sin(phi1)
        Fy_total[mask] = -F_c_total * torch.sin(phi1) + F_t_total * torch.cos(phi1)

    # VB > VB_star (Elastic + plastic)
    if mask_inv.any():
        VB2 = VB[mask_inv]
        mu2 = mu[mask_inv]
        Fc2 = Fc[mask_inv]
        Ft2 = Ft[mask_inv]
        shearangle2 = shearangle[mask_inv]
        k2 = k[mask_inv]
        phi2 = phi[mask_inv]

        ita_w = 0.5 * torch.arccos(torch.tensor(0.99, device=device))
        tau_0 = k2 * torch.cos(2 * ita_w)
        sigma_0 = k2 * (1 + torch.pi / 2 + 2 * ita_w + torch.sin(2 * ita_w))

        # Integration results
        F_tw_2 = sigma_0 * (VB2 - 2 * VB_star / 3) * axial_depth
        
        x_t = VB2 - VB_star * torch.sqrt(tau_0 / sigma_0)
        F_cw_2 = (tau_0 * x_t + mu2 * sigma_0 * (VB_star / 3) * (tau_0 / sigma_0)**(3/2)) * axial_depth
        
        F_c_total = Fc2 + F_cw_2
        F_t_total = Ft2 + F_tw_2

        Fx_total[mask_inv] = F_c_total * torch.cos(phi2) + F_t_total * torch.sin(phi2)
        Fy_total[mask_inv] = -F_c_total * torch.sin(phi2) + F_t_total * torch.cos(phi2)

    return Fx_total, Fy_total
    