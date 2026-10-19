"""Shared pieces for the geophysics lessons: magnetic physics, priors, MCMC, flow builders.

Coordinates: x runs along the survey line (towards magnetic north), z points DOWN.
Sensors fly at height `h` above flat ground (z = -h). Units: metres, nanotesla (nT), SI susceptibility.
"""
import math

import torch
import normflows as nf


# ---------------------------------------------------------------------------
# Physics: total-field magnetic anomaly of a buried, magnetised sphere
# ---------------------------------------------------------------------------

def sphere_tmi(x_sensors, x0, depth, radius, chi, F=50000.0, inc_deg=-50.0, h=60.0):
    """Total-field anomaly (nT) of a sphere magnetised by the Earth's field.

    A uniformly magnetised sphere looks, from outside, exactly like a tiny bar magnet (a dipole) at its centre.
    Its strength is (volume x susceptibility x Earth's field); its direction follows the Earth's field.

    Args:
      x_sensors: (n_sensors,) positions along the line
      x0, depth, radius, chi: (batch,) sphere centre position, centre depth below ground, radius, susceptibility
      F: Earth's field strength in nT (about 50,000 nT in northern Australia)
      inc_deg: field inclination; negative = field points upward (southern hemisphere)
      h: sensor height above ground

    Returns:
      (batch, n_sensors) anomaly in nT
    """
    inc = math.radians(inc_deg)
    fx, fz = math.cos(inc), math.sin(inc)            # unit vector of Earth's field (x, z-down)
    rx = x_sensors[None, :] - x0[:, None]             # vector from sphere centre to each sensor
    rz = -h - depth[:, None]
    r = torch.sqrt(rx**2 + rz**2)
    cos_angle = (rx * fx + rz * fz) / r               # angle between field and sensor direction
    strength = (radius**3 * chi * F / 3.0)[:, None]   # "size of the magnet" (nT * m^3)
    return strength * (3 * cos_angle**2 - 1) / r**3


# ---------------------------------------------------------------------------
# Parameters: the flow works on 4 unitless numbers theta; these map them to physical values.
# The prior ("what we believe before measuring") is simply standard normal on theta.
# ---------------------------------------------------------------------------

PARAM_NAMES = ["x0 (m)", "depth (m)", "radius (m)", "susceptibility (SI)"]
# theta_i = (transform(value) - centre) / spread
X0_SCALE = 300.0                   # x0 ~ N(0, 300 m)
LOG_DEPTH = (math.log(150.0), 0.5) # depth ~ 150 m, give or take a factor of ~1.6
LOG_RADIUS = (math.log(30.0), 0.5) # radius ~ 30 m
LOG_CHI = (math.log(0.05), 1.0)    # susceptibility ~ 0.05 SI, give or take a factor of ~2.7


def theta_to_phys(theta):
    """(batch, 4) unitless -> dict of physical tensors."""
    return {
        "x0": theta[:, 0] * X0_SCALE,
        "depth": torch.exp(LOG_DEPTH[0] + LOG_DEPTH[1] * theta[:, 1]),
        "radius": torch.exp(LOG_RADIUS[0] + LOG_RADIUS[1] * theta[:, 2]),
        "chi": torch.exp(LOG_CHI[0] + LOG_CHI[1] * theta[:, 3]),
    }


def phys_to_theta(x0, depth, radius, chi):
    """Physical values (floats or tensors) -> (batch, 4) theta."""
    t = lambda v: torch.as_tensor(v, dtype=torch.get_default_dtype()).reshape(-1)
    return torch.stack([
        t(x0) / X0_SCALE,
        (torch.log(t(depth)) - LOG_DEPTH[0]) / LOG_DEPTH[1],
        (torch.log(t(radius)) - LOG_RADIUS[0]) / LOG_RADIUS[1],
        (torch.log(t(chi)) - LOG_CHI[0]) / LOG_CHI[1],
    ], dim=1)


def phys_matrix(theta):
    """(batch, 4) theta -> (batch, 4) physical values in PARAM_NAMES order (for plotting/tables)."""
    p = theta_to_phys(theta)
    return torch.stack([p["x0"], p["depth"], p["radius"], p["chi"]], dim=1)


def predict(theta, x_sensors, **physics_kw):
    p = theta_to_phys(theta)
    return sphere_tmi(x_sensors, p["x0"], p["depth"], p["radius"], p["chi"], **physics_kw)


# ---------------------------------------------------------------------------
# Simulator for Recipe A (conditional flow): invent undergrounds, compute their readings
# ---------------------------------------------------------------------------

def simulate(n, x_sensors, noise, generator=None, **physics_kw):
    """Draw n undergrounds from the prior and the noisy readings they would produce.

    Returns theta (n, 4) and readings (n, n_sensors)."""
    theta = torch.randn(n, 4, generator=generator)
    clean = predict(theta, x_sensors, **physics_kw)
    return theta, clean + noise * torch.randn(clean.shape, generator=generator)


def readings_to_context(readings, noise):
    """What the conditional flow 'sees': readings in noise units, squashed with asinh.

    Shallow strong lumps can give hundreds of nT and deep weak ones a few nT; asinh keeps small values
    nearly unchanged (asinh(v) ~ v) while compressing large ones (~ log), so the network sees both well."""
    return torch.asinh(readings / noise)


# ---------------------------------------------------------------------------
# The score (unnormalised log posterior) used by Recipe B
# ---------------------------------------------------------------------------

class SpherePosterior(nf.distributions.Target):
    """log p(theta | readings) up to a constant = log prior + log likelihood.

    - prior: standard normal on theta ("does it make sense?")
    - likelihood: readings = physics(theta) + Gaussian noise with std `noise` nT ("does it match?")
    """

    def __init__(self, x_sensors, readings, noise, **physics_kw):
        super().__init__()
        self.register_buffer("x_sensors", x_sensors)
        self.register_buffer("readings", readings)
        self.noise = noise
        self.physics_kw = physics_kw

    def log_prob(self, theta):
        pred = predict(theta, self.x_sensors, **self.physics_kw)
        log_lik = -0.5 * (((pred - self.readings) / self.noise) ** 2).sum(-1)
        log_prior = -0.5 * (theta**2).sum(-1)
        return log_lik + log_prior


# ---------------------------------------------------------------------------
# Flow builder: neural spline flow (same family as the README example)
# ---------------------------------------------------------------------------

def make_spline_flow(dim, n_layers=6, hidden=64, target=None, context_dim=None, coupled=True):
    """Spline flow. `coupled=True` (half the numbers steer the other half) is fast in both directions;
    `coupled=False` (autoregressive, one number at a time) is a bit more flexible but slow to sample."""
    layer = nf.flows.CoupledRationalQuadraticSpline if coupled else nf.flows.AutoregressiveRationalQuadraticSpline
    flows = []
    for _ in range(n_layers):
        flows.append(layer(dim, 2, hidden, num_context_channels=context_dim))
        flows.append(nf.flows.LULinearPermute(dim))
    q0 = nf.distributions.DiagGaussian(dim, trainable=False)
    if context_dim is None:
        return nf.NormalizingFlow(q0=q0, flows=flows, p=target)
    return nf.ConditionalNormalizingFlow(q0=q0, flows=flows, p=target)


# ---------------------------------------------------------------------------
# Gold standard: plain random-walk Metropolis MCMC, many chains in parallel
# ---------------------------------------------------------------------------

def find_map(log_prob, dim, n_starts=64, steps=2000, seed=0):
    """Best-scoring theta from many random starts (used to start the MCMC chains)."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(n_starts, dim, generator=g).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        (-log_prob(theta).sum()).backward()
        opt.step()
    with torch.no_grad():
        lp = log_prob(theta)
    return theta.detach()[lp.argmax()]


def run_mcmc(log_prob, init, n_chains=256, n_burn=4000, n_keep=4000, thin=4, seed=0):
    """Random-walk Metropolis with the proposal shape adapted during burn-in.

    Slow but trustworthy for a 4-number problem: given enough steps it samples the exact posterior.
    Returns (samples (n_chains * n_keep/thin, dim), acceptance rate, R-hat per dimension).
    """
    g = torch.Generator().manual_seed(seed)
    dim = init.numel()
    x = init[None, :] + 0.01 * torch.randn(n_chains, dim, generator=g)
    lp = log_prob(x)
    chol = 0.05 * torch.eye(dim)
    kept, n_acc, n_prop = [], 0, 0
    history = []
    for it in range(n_burn + n_keep):
        prop = x + torch.randn(n_chains, dim, generator=g) @ chol.T
        lp_prop = log_prob(prop)
        accept = torch.rand(n_chains, generator=g).log() < (lp_prop - lp)
        x = torch.where(accept[:, None], prop, x)
        lp = torch.where(accept, lp_prop, lp)
        if it < n_burn:
            history.append(x.clone())
            if (it + 1) % 500 == 0:  # adapt: proposal covariance = (2.38^2/d) * current spread of the chains
                recent = torch.cat(history[-250:])
                cov = torch.cov(recent.T) * (2.38**2 / dim) + 1e-9 * torch.eye(dim)
                chol = torch.linalg.cholesky(cov)
        else:
            n_acc += accept.sum().item(); n_prop += n_chains
            if (it - n_burn) % thin == 0:
                kept.append(x.clone())
    chains = torch.stack(kept, dim=1)  # (n_chains, n_kept, dim)
    # Gelman-Rubin R-hat: ~1.00 means the chains agree with each other
    n = chains.shape[1]
    w = chains.var(dim=1).mean(0)
    b = n * chains.mean(1).var(0)
    rhat = torch.sqrt(((n - 1) / n * w + b / n) / w)
    return chains.reshape(-1, dim), n_acc / n_prop, rhat


def importance_ess(log_p, log_q):
    """Effective sample size fraction of importance weights p/q (1.0 = flow is perfect)."""
    log_w = log_p - log_q
    log_w = log_w - log_w.max()
    w = log_w.exp()
    return ((w.sum() ** 2) / (w**2).sum() / len(w)).item()
