"""Lesson 4 helpers: Osborne airborne magnetic data, a magnetised rectangular block (prism), priors and the score.

Coordinates: metres relative to the Osborne mine's listed position, x = north, y = east, z = DOWN
(the convention of Blakely, 1995, "Potential Theory in Gravity and Magnetic Applications").
"""
import math

import numpy as np
import pandas as pd
import torch
import normflows as nf

# Osborne mine, listed position (Wikidata / Mindat): 22°05'47"S, 140°34'52"E.
MINE_LAT = -(22 + 5 / 60 + 47 / 3600)
MINE_LON = 140 + 34 / 60 + 52 / 3600
# Earth's field at Osborne in mid-1990 (IGRF, computed with ppigrf)
F_NT, INC, DEC = 52086.0, -53.4, 6.7
TERRAIN_CLEARANCE = 80.0  # survey flown ~80 m above ground
MU0 = 4 * math.pi * 1e-7

TRAIN_LINES = [5671, 5672, 5674, 5675, 5676, 5678, 5679, 5681]
HOLDOUT_LINES = [5673, 5677, 5680]


def load_window(path, north=(-1300, 1500), east=(-1900, 500), every=5):
    """Readings in a window around the mine, as local coordinates. Keeps east-west lines only, every `every`-th reading."""
    df = pd.read_csv(path)
    df["north"] = (df.latitude - MINE_LAT) * 110574.0
    df["east"] = (df.longitude - MINE_LON) * 111320.0 * math.cos(math.radians(MINE_LAT))
    w = df[df.north.between(*north) & df.east.between(*east)]
    w = w[w.flight_line.isin(TRAIN_LINES + HOLDOUT_LINES)]
    w = w.sort_values(["flight_line", "east"])
    w = w[w.groupby("flight_line").cumcount() % every == 0]
    # The plane followed the terrain at ~80 m clearance, so every sensor sits ~80 m above the ground beneath it;
    # depths in the model are measured below that (locally flat) ground surface.
    w = w.assign(z=-TERRAIN_CLEARANCE)
    return w.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Physics: total-field anomaly of a magnetised rectangular prism
# ---------------------------------------------------------------------------

def _mbox(al1, al2, be1, be2, h, fm):
    """Blakely (1995) 'mbox': prism from depth h (below the sensor) to infinity, sides at al1..al2, be1..be2.
    All tensors broadcast to (batch, n_obs). fm = direction-cosine products (fm1..fm6)."""
    fm1, fm2, fm3, fm4, fm5, fm6 = fm
    t = 0.0
    for i, al in enumerate((al1, al2)):
        for j, be in enumerate((be1, be2)):
            sign = 1.0 if i == j else -1.0
            r0 = torch.sqrt(al**2 + be**2 + h**2)
            r0h = r0 * h
            ab = al * be
            # log((r0-al)/(r0+al))/2 written in a numerically stable form: -asinh(al / sqrt(be^2 + h^2))
            tlog = (-fm3 * torch.asinh(al / torch.sqrt(be**2 + h**2))
                    - fm2 * torch.asinh(be / torch.sqrt(al**2 + h**2))
                    - fm1 * torch.log(r0 + h))
            tatan = (-fm4 * torch.atan2(ab, al**2 + r0h + h**2)
                     - fm5 * torch.atan2(ab, r0**2 + r0h - al**2)
                     + fm6 * torch.atan2(ab, r0h))
            t = t + sign * (tlog + tatan)
    return t


def prism_tmi(obs_n, obs_e, obs_z, xc, yc, top, bottom, length, width, strike, chi,
              F=F_NT, inc=INC, dec=DEC):
    """Total-field anomaly (nT) of a vertical-sided block, magnetised by the Earth's field (induced only,
    no self-demagnetisation, no remanence).

    obs_*: (n_obs,) sensor positions. Block parameters: (batch,) tensors.
      xc, yc: centre (north, east); top, bottom: depths of top and bottom faces below ground;
      length/width: horizontal size along / across strike; strike: azimuth of the long axis (radians, clockwise from north).
    """
    xc, yc, top, bottom, length, width, strike, chi = (v[:, None] for v in (xc, yc, top, bottom, length, width, strike, chi))
    dn, de = obs_n[None, :] - xc, obs_e[None, :] - yc
    c, s = torch.cos(strike), torch.sin(strike)
    u = dn * c + de * s          # sensor position along strike, relative to block centre
    v = -dn * s + de * c         # ... across strike
    # direction cosines of the field in the rotated frame (magnetisation is parallel to the field)
    i, d = math.radians(inc), math.radians(dec)
    a = math.cos(i) * torch.cos(d - strike)
    b = math.cos(i) * torch.sin(d - strike)
    cc = math.sin(i) * torch.ones_like(a)
    fm = (2 * a * b, 2 * a * cc, 2 * b * cc, a * a, b * b, cc * cc)
    al1, al2 = -length / 2 - u, length / 2 - u
    be1, be2 = -width / 2 - v, width / 2 - v
    zobs = obs_z[None, :]
    t = _mbox(al1, al2, be1, be2, top - zobs, fm) - _mbox(al1, al2, be1, be2, bottom - zobs, fm)
    M = chi * F * 1e-9 / MU0      # induced magnetisation (A/m)
    return t * M * 1e-7 * 1e9     # (mu0/4pi) * ... in tesla -> nT


# ---------------------------------------------------------------------------
# Parameters: 9 unitless numbers per block, + 3 for the regional background, + 1 misfit level.
# The prior is standard normal on every one of them.
# ---------------------------------------------------------------------------

BLOCK_PARAMS = 9
CENTRE0 = (300.0, -700.0)  # rough middle of the anomaly (north, east): just where the prior is centred


def dim(n_blocks):
    return BLOCK_PARAMS * n_blocks + 4


def theta_to_phys(theta, n_blocks):
    """(batch, dim) -> dict. Block quantities are (batch, n_blocks); prior ~90% ranges in comments.

    Blocks are ordered west -> east: block k+1's centre is 100-1700 m east of block k's, so 'block 1' and
    'block 2' can't swap labels (swapping would give two identical mirror-image answers, which flows handle badly)."""
    B = theta.shape[0]
    blk = theta[:, : BLOCK_PARAMS * n_blocks].reshape(B, n_blocks, BLOCK_PARAMS)
    rest = theta[:, BLOCK_PARAMS * n_blocks:]
    north = CENTRE0[0] + 800.0 * blk[..., 0]                          # ±1.3 km
    east_first = CENTRE0[1] + 800.0 * blk[:, :1, 1]
    east_steps = torch.exp(math.log(400.0) + 0.8 * blk[:, 1:, 1])     # 110 … 1500 m between neighbours
    east = torch.cat([east_first, east_first + torch.cumsum(east_steps, dim=1)], dim=1)
    return {
        "xc": north, "yc": east,
        "top": torch.exp(math.log(80.0) + 1.0 * blk[..., 2]),           # 15 … 410 m below ground
        "extent": torch.exp(math.log(300.0) + 1.0 * blk[..., 3]),       # 60 … 1550 m tall
        "length": torch.exp(math.log(500.0) + 0.8 * blk[..., 4]),       # 130 … 1870 m
        "width": torch.exp(math.log(150.0) + 0.8 * blk[..., 5]),        # 40 … 560 m
        "strike": 0.5 * torch.atan2(blk[..., 7], blk[..., 6]),          # (u, v) -> angle, uniform 0 … 180°
        "chi": torch.exp(math.log(0.3) + 1.5 * blk[..., 8]),            # 0.03 … 3.5 SI (magnetite ironstone can be very magnetic)
        "c0": 400.0 + 300.0 * rest[:, 0],                                # regional background level (nT)
        "gn": 0.2 * rest[:, 1], "ge": 0.2 * rest[:, 2],                 # regional slope (nT per m)
        "sigma": torch.exp(math.log(30.0) + 1.0 * rest[:, 3]),          # 6 … 155 nT: how well the model explains the data
    }


def predict(theta, n_blocks, obs_n, obs_e, obs_z):
    p = theta_to_phys(theta, n_blocks)
    total = p["c0"][:, None] + p["gn"][:, None] * obs_n[None, :] + p["ge"][:, None] * obs_e[None, :]
    for k in range(n_blocks):
        total = total + prism_tmi(obs_n, obs_e, obs_z, p["xc"][:, k], p["yc"][:, k], p["top"][:, k],
                                  p["top"][:, k] + p["extent"][:, k], p["length"][:, k], p["width"][:, k],
                                  p["strike"][:, k], p["chi"][:, k])
    return total


class OsbornePosterior(nf.distributions.Target):
    """log prior + log likelihood. The misfit level sigma is learned from the data: it absorbs instrument noise
    *and* everything the blocks can't explain. A big sigma is the model telling you it's too simple."""

    def __init__(self, n_blocks, obs_n, obs_e, obs_z, readings, physics_device="cpu", temper=1.0):
        super().__init__()
        self.n_blocks = n_blocks
        # temper < 1 down-weights the evidence: use when neighbouring misfits are correlated, so the readings
        # are worth fewer independent pieces of evidence than their count suggests (Lesson 5)
        self.temper = temper
        self.physics_device = physics_device  # the block physics is heavy: run it on the GPU if there is one
        self.data = {k: v.to(physics_device) for k, v in dict(obs_n=obs_n, obs_e=obs_e, obs_z=obs_z, readings=readings).items()}

    def log_prob(self, theta):
        th = theta.to(self.physics_device)
        sigma = theta_to_phys(th, self.n_blocks)["sigma"]
        d = self.data
        pred = predict(th, self.n_blocks, d["obs_n"], d["obs_e"], d["obs_z"])
        n = d["readings"].numel()
        log_lik = -0.5 * (((pred - d["readings"]) / sigma[:, None]) ** 2).sum(-1) - n * torch.log(sigma)
        return (self.temper * log_lik - 0.5 * (th**2).sum(-1)).to(theta.device)


def find_map(post, n_starts=64, steps=2000, seed=0, init=None):
    """Best-scoring theta from many random starts (plus optional given starts)."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(n_starts, dim(post.n_blocks), generator=g)
    if init is not None:
        theta = torch.cat([init, theta])
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.03)
    for _ in range(steps):
        opt.zero_grad()
        (-post.log_prob(theta)).sum().backward()
        opt.step()
    with torch.no_grad():
        lp = post.log_prob(theta)
    order = lp.argsort(descending=True)
    return theta.detach()[order], lp[order]


def train_flow(post, theta_map, n_iter, seed=0, log_every=500):
    """Warm-started Recipe B training (the loop used in Lessons 4 and 5)."""
    import time
    scale = laplace_scale(post, theta_map)
    torch.manual_seed(seed)
    flow = warm_started_flow(theta_map, scale)
    opt = torch.optim.Adam(flow.parameters(), lr=5e-4)
    losses, t0 = [], time.time()
    for it in range(n_iter):
        theta, log_q = flow.sample(256)
        loss = (log_q - post.log_prob(theta)).mean()
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0); opt.step()
        losses.append(loss.item())
        if (it + 1) % log_every == 0:
            print(f"  step {it+1:5d}  loss {np.mean(losses[-100:]):9.2f}   [{time.time()-t0:.0f}s]")
    flow.eval()
    return flow, scale, losses


def laplace_scale(post, theta_map, floor=1e-3, cap=1.0):
    """Rough posterior spread per parameter from the curvature of the score at the best fit (Laplace approximation)."""
    H = torch.autograd.functional.hessian(lambda t: -post.log_prob(t[None])[0], theta_map)
    evals, evecs = torch.linalg.eigh(0.5 * (H + H.T))
    cov = evecs @ torch.diag(1.0 / evals.clamp(min=1.0)) @ evecs.T   # clamp: flat directions -> spread of at most the prior
    return cov.diagonal().sqrt().clamp(floor, cap)


def warm_started_flow(theta_map, scale, n_layers=6, hidden=128):
    """Spline flow that works in 'standardised' coordinates around the best fit:
        dice -> spline/mixing layers -> eps -> theta = theta_map + scale * eps   (last layer, also trainable)
    On real data the posterior is a needle in a haystack; starting the flow on the needle makes training far easier."""
    d = theta_map.numel()
    flows = []
    for _ in range(n_layers):
        flows.append(nf.flows.CoupledRationalQuadraticSpline(d, 2, hidden))
        flows.append(nf.flows.LULinearPermute(d))
    shift_scale = nf.flows.AffineConstFlow((d,))
    with torch.no_grad():
        shift_scale.s.copy_(torch.log(scale).reshape(1, -1))
        shift_scale.t.copy_(theta_map.reshape(1, -1))
    flows.append(shift_scale)
    return nf.NormalizingFlow(q0=nf.distributions.DiagGaussian(d, trainable=False), flows=flows)


def phys_table(theta, n_blocks):
    """DataFrame of interpretable quantities, one row per sample."""
    p = theta_to_phys(theta, n_blocks)
    cols = {}
    for k in range(n_blocks):
        tag = f"B{k+1} " if n_blocks > 1 else ""
        cols.update({
            f"{tag}north (m)": p["xc"][:, k], f"{tag}east (m)": p["yc"][:, k], f"{tag}top (m)": p["top"][:, k],
            f"{tag}bottom (m)": p["top"][:, k] + p["extent"][:, k], f"{tag}length (m)": p["length"][:, k],
            f"{tag}width (m)": p["width"][:, k], f"{tag}strike (deg)": torch.rad2deg(p["strike"][:, k]) % 180,
            f"{tag}chi (SI)": p["chi"][:, k],
        })
    cols["misfit level (nT)"] = p["sigma"]
    return pd.DataFrame({k: v.detach().cpu().numpy() for k, v in cols.items()})
