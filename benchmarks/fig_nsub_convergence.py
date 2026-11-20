"""Convergence of the *derivatives* with the PREM sub-shell refinement n_sub.

The paper's Validation section shows that the forward probability converges as
n_sub grows; this script shows the same for the quantities that the worked
Earth-tomography example (section "A worked example") is built from, and for a
geometry derivative either side of the core-crossing threshold:

(a) the Fisher outputs of ``demo_tomography_fisher.py`` -- marginalized and
    fixed sigma of the bulk core, the marginalized inner-core sigma, the
    inner/outer-core correlation, and the design derivative
    d sigma / d sigma_cz -- re-evaluated at each n_sub.  That example uses the
    parametric ``LayeredEarth`` (one density per shell, PREM at the shell
    mid-radius), not the default path-midpoint PREM path, so it is checked on
    its own model, against its own finest refinement (n_sub = FISHER_REF);
(b) the per-bin count derivative d mu / d ln rho_core,bulk (panel (a) of the
    tomography figure) against the same reference;
(c,d) the geometry derivative dP(nu_mu -> nu_mu)/d cos(zenith) through the
    PREM path (the one of the density-derivative figure) on a dense zenith grid
    either side of the core-crossing threshold, for both ways of cutting the
    chord (``segmentation=`` in ``mango.probability_earth``):
      "radial" (default)  -- at every sub-shell boundary.  Each sub-shell radius
                             is an angle at which the chord grazes it, so the
                             model has a square-root cusp in dP/dcz there; the
                             cusps shrink with n_sub but their number grows.
      "chord"             -- each PREM region's chord interval in equal pieces,
                             same count.  No artificial cusps; converges as
                             1/n_sub^2.

A third figure (mango_nsub_paths) repeats the tomography example of (a,b)
on all three Earth paths of the demo (its EARTH_PATH switch): the LayeredEarth
of the paper, and the PREM polynomial path cut radially or in chord pieces,
each with the same per-zone density scales.  All three are measured against
one common limit, the chord path at n_sub = FISHER_REF, so the figure also
shows that the three discretizations converge to the same answer.

The reference for (c,d) is ``segmentation="chord"`` at n_sub = GEOM_REF; its own
uncertainty, estimated from n_sub = GEOM_REF/2 (Richardson, second order), is
quoted in (d).  A finer *radial* table cannot serve: at n_sub = 256 its grazing
cusps are ~1e-4 apart in cos(zenith) and a residual against it is dominated by
the reference's own spikes.  (tests/test_chord_segmentation.py checks the chord
answer independently against the continuous-profile ODE backend.)

The physical core-mantle cusp, where the derivative itself diverges, is
excluded from the summary statistics within +-CORE_EXCL; the shell-grazing
cusps of the radial cut are not excluded -- they are what the median and 95th
percentile in (d) are measuring.

Usage (from the repository root):
    JAX_PLATFORMS=cpu python benchmarks/fig_nsub_convergence.py          # compute + plot
    JAX_PLATFORMS=cpu python benchmarks/fig_nsub_convergence.py --plot   # replot cached JSON
Takes ~15 minutes on CPU.  Results are cached item by item in
benchmarks/output/mango_nsub_convergence.json, so an interrupted run resumes;
delete the JSON to recompute from scratch.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "output")
os.makedirs(OUTDIR, exist_ok=True)
CACHE = os.path.join(OUTDIR, "mango_nsub_convergence.json")

FISHER_NSUB = [1, 2, 4, 8, 16, 32]
FISHER_REF = 128
FISHER_PATHS = ("layered", "radial", "chord")   # EARTH_PATH of the demo
GEOM_NSUB = [1, 2, 4, 8, 16, 32, 64]
GEOM_REF = 256                               # chord-segmentation reference
SCHEMES = ("radial", "chord")
GEOM_E = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 15.0, 20.0])   # GeV
GEOM_CZ = np.linspace(-0.90, -0.77, 1301)
GEOM_SHOW_E = 6.0                            # energy drawn in panel (c)
SHOW_NSUB = [4, 16]
FIG6A_SLICE_E = (1.2, 2.9)                   # GeV rows of Fig. 6a shown as slices
FIG6A_SLICE_NSUB = [1, 2, 4]
FIG6A_DIFF_NSUB = [1, 4, 16]
CORE_CZ = -0.8376384419020163
CORE_EXCL = 3e-3
H_ATM_KM = 15.0
CZ_CHUNK = 128                               # angles per batched call


# --------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------
def _fisher_worker(n_sub, path="layered"):
    """Run in a subprocess: the demo fixes its Earth model at import."""
    os.environ["N_SUB"] = str(n_sub)
    os.environ["EARTH_PATH"] = path
    sys.path.insert(0, HERE)
    import jax
    import jax.numpy as jnp
    import demo_tomography_fisher as T

    F, mu, J = T.fisher(T.THETA0)
    Finv = jnp.linalg.inv(F)
    v = T.BULK_CORE_VEC
    nv = float(T._bulk_norm(v))
    out = {
        "n_sub": n_sub,
        "path": path,
        "n_shell": int(T._BASE_EARTH.outer.size),
        "sigma_bulk_marg": float(jnp.sqrt(v @ Finv @ v) / nv),
        "sigma_bulk_fixed": float(1.0 / jnp.sqrt(v @ F @ v)),
        "sigma_ic_marg": float(jnp.sqrt(Finv[0, 0])),
        "corr_ic_oc": float(Finv[0, 1] / jnp.sqrt(Finv[0, 0] * Finv[1, 1])),
        "dsigma_dres": float(jax.grad(T.sigma_core)(T.RES_CZ)),
        "dmu_dcore": np.asarray(J @ v).tolist(),
    }
    print(json.dumps(out))


def _load_cache():
    data = {"fisher": [], "geometry": {}}
    if os.path.exists(CACHE):
        with open(CACHE) as fh:
            data = json.load(fh)
    for r in data["fisher"]:            # entries from before EARTH_PATH existed
        r.setdefault("path", "layered")
    # geometry entries are only reusable on the same (E, cz) grid
    if (data.get("geom_E") != GEOM_E.tolist()
            or data.get("geom_cz") != GEOM_CZ.tolist()):
        data["geometry"] = {}
    return data


def _save_cache(data):
    data["geom_E"], data["geom_cz"] = GEOM_E.tolist(), GEOM_CZ.tolist()
    with open(CACHE + ".tmp", "w") as fh:
        json.dump(data, fh)
    os.replace(CACHE + ".tmp", CACHE)


def compute_fisher(data):
    done = {(r["path"], r["n_sub"]) for r in data["fisher"]}
    for path in FISHER_PATHS:
        for n in FISHER_NSUB + [FISHER_REF]:
            if (path, n) in done:
                continue
            res = subprocess.run(
                [sys.executable, __file__, "--fisher-worker", str(n), path],
                check=True, capture_output=True, text=True)
            data["fisher"].append(json.loads(res.stdout.strip().splitlines()[-1]))
            _save_cache(data)
            print(f"  fisher {path:7s} n_sub={n:4d}: "
                  f"sigma_marg={data['fisher'][-1]['sigma_bulk_marg']:.5f}",
                  flush=True)


def _fisher_by_n(data, path="layered"):
    return {r["n_sub"]: r for r in data["fisher"] if r["path"] == path}


def _dp_dcz_grid(prob):
    import jax
    import jax.numpy as jnp

    g = jax.jit(jax.vmap(jax.grad(prob), in_axes=(0, None)))
    cz = jnp.asarray(GEOM_CZ)
    # one energy and at most CZ_CHUNK angles per call: batching the whole
    # (E, cz) grid through thousands of segments exhausts 64 GB
    return np.asarray([np.concatenate([np.asarray(g(cz[i:i + CZ_CHUNK], float(e)))
                                       for i in range(0, cz.size, CZ_CHUNK)])
                       for e in GEOM_E])


def compute_geometry(data):
    import mango
    import mango.earth as earth

    p0 = mango.nufit_no()
    out = data["geometry"]
    runs = [(seg, n) for seg in SCHEMES for n in GEOM_NSUB]
    runs += [("chord", GEOM_REF // 2), ("chord", GEOM_REF)]
    for seg, n in runs:
        key = f"{seg}{n}"
        if key in out:
            continue
        prob = lambda c, e, n=n, seg=seg: mango.probability_earth(
            p0, e, c, n_sub=n, h_atm_km=H_ATM_KM, segmentation=seg)[1, 1]
        crit = earth.critical_cos_zenith(n_sub=n, segmentation=seg)
        out[key] = {
            "dP_dcz": _dp_dcz_grid(prob).tolist(),
            "critical_cz": [float(c) for c in crit if GEOM_CZ[0] < c < GEOM_CZ[-1]],
        }
        _save_cache(data)
        print(f"  geometry {seg:6s} n_sub={n:4d} done", flush=True)


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------
def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})
    fish = _fisher_by_n(data)
    geo = data["geometry"]
    ref_f = fish[FISHER_REF]
    dref = np.asarray(geo[f"chord{GEOM_REF}"]["dP_dcz"])
    dref_half = np.asarray(geo[f"chord{GEOM_REF // 2}"]["dP_dcz"])
    ref_err = np.abs(dref - dref_half) / 3.0       # Richardson, 2nd order

    fig = plt.figure(figsize=(12.5, 9.6))
    outer = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.15], hspace=0.32,
                     wspace=0.28)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 0],
                                    height_ratios=[1.3, 1], hspace=0.06)
    ax_c = fig.add_subplot(inner[0])
    ax_cr = fig.add_subplot(inner[1], sharex=ax_c)
    ax_d = fig.add_subplot(outer[1, 1])

    def mark_fig6(ax):
        ax.axvline(4, color="0.55", ls=":", lw=1)

    def guide(ax, x0, y0, lo, hi, label_offset=0.35, xm=None):
        n = np.array([lo, hi], dtype=float)
        ax.loglog(n, y0 * (n / x0) ** -2, color="0.6", lw=0.8)
        xm = np.sqrt(lo * hi) if xm is None else xm
        ax.text(xm, y0 * (xm / x0) ** -2 * label_offset, r"$\propto n_{\rm sub}^{-2}$",
                color="0.45", fontsize=9)

    # (a) Fisher outputs -------------------------------------------------------
    quantities = [
        ("sigma_bulk_marg", r"$\sigma(\ln\rho_{\rm core,bulk})$, marginalized", "o"),
        ("sigma_bulk_fixed", r"$\sigma(\ln\rho_{\rm core,bulk})$, others fixed", "s"),
        ("sigma_ic_marg", r"$\sigma(\ln\rho_{\rm inner\ core})$, marginalized", "D"),
        ("corr_ic_oc", r"corr(inner, outer core)", "^"),
        ("dsigma_dres", r"$\partial\sigma/\partial\sigma_{\cos\theta_z}$", "v"),
    ]
    for key, lab, m in quantities:
        rel = [abs(fish[n][key] / ref_f[key] - 1.0) for n in FISHER_NSUB]
        ax_a.loglog(FISHER_NSUB, rel, marker=m, ms=5, lw=1.1, label=lab)
    mark_fig6(ax_a)
    ax_a.set_xlabel(r"$n_{\rm sub}$")
    ax_a.set_ylabel(fr"relative deviation from $n_{{\rm sub}}={FISHER_REF}$")
    ax_a.set_title("(a) Fisher outputs of the tomography example", loc="left",
                   fontsize=11)
    ax_a.legend(fontsize=8.5, frameon=False, loc="lower left")

    # (b) per-bin count derivative --------------------------------------------
    jref = np.asarray(ref_f["dmu_dcore"])
    scale = np.abs(jref).max()
    informative = np.abs(jref) > 1e-2 * scale
    dev = [np.abs(np.asarray(fish[n]["dmu_dcore"]) - jref) / scale
           for n in FISHER_NSUB]
    ax_b.loglog(FISHER_NSUB, [x.max() for x in dev], "o-", label="maximum, all bins")
    ax_b.loglog(FISHER_NSUB, [np.percentile(x[informative], 95) for x in dev],
                "s--", label=r"95th percentile, bins with $|\partial\mu_b|>1\%$ of max")
    ax_b.loglog(FISHER_NSUB, [np.median(x[informative]) for x in dev], "D:",
                label="median, same bins")
    guide(ax_b, 1.0, 1.0e-2, 1, 32)
    mark_fig6(ax_b)
    ax_b.set_xlabel(r"$n_{\rm sub}$")
    ax_b.set_ylabel(r"$|\Delta\,\partial\mu_b/\partial\ln\rho_{\rm core,bulk}|"
                    r"\,/\,\max_b|\partial\mu_b/\partial\ln\rho_{\rm core,bulk}|$")
    ax_b.set_title(r"(b) per-bin $\partial\mu_b/\partial\ln\rho_{\rm core,bulk}$"
                   " (Fig. 6a)", loc="left", fontsize=11)
    ax_b.legend(fontsize=8.5, frameon=False, loc="lower left")

    # (c) dP/dcz across the core threshold ------------------------------------
    ie = int(np.argmin(np.abs(GEOM_E - GEOM_SHOW_E)))
    style = {"radial": dict(color="C3", label="radial sub-shells (default)"),
             "chord": dict(color="C0", label="chord pieces")}
    shade = dict(zip(SHOW_NSUB, [1.0, 0.45]))
    ax_c.plot(GEOM_CZ, dref[ie], color="k", lw=2.6, alpha=0.25, label="reference")
    for seg in SCHEMES:
        n = SHOW_NSUB[0]
        d = np.asarray(geo[f"{seg}{n}"]["dP_dcz"])[ie]
        ax_c.plot(GEOM_CZ, d, color=style[seg]["color"], lw=0.9,
                  label=fr"{style[seg]['label']}, $n_{{\rm sub}}={n}$")
        for n in SHOW_NSUB:
            d = np.asarray(geo[f"{seg}{n}"]["dP_dcz"])[ie]
            ax_cr.semilogy(GEOM_CZ, np.abs(d - dref[ie]) + 1e-12,
                           color=style[seg]["color"], alpha=shade[n], lw=0.7)
    ax_cr.semilogy(GEOM_CZ, ref_err[ie] + 1e-12, color="k", lw=0.7, alpha=0.5)
    for cc in geo[f"radial{SHOW_NSUB[0]}"]["critical_cz"]:
        ax_cr.axvline(cc, color="C3", ls=":", lw=0.6)
    for a in (ax_c, ax_cr):
        a.axvspan(CORE_CZ - CORE_EXCL, CORE_CZ + CORE_EXCL, color="0.5", alpha=0.15,
                  lw=0)
    ax_c.set_ylim(-27, 45)
    ax_c.set_ylabel(r"$\partial P_{\mu\mu}/\partial\cos\theta_z$")
    ax_c.set_title(fr"(c) geometry derivative at $E_\nu={GEOM_E[ie]:g}$ GeV",
                   loc="left", fontsize=11)
    ax_c.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax_c.text(CORE_CZ + 0.004, 18, "core threshold", color="0.35", fontsize=8.5,
              va="top")
    ax_c.text(GEOM_CZ[-1] - 0.001, -25,
              fr"residuals below: dark $n_{{\rm sub}}={SHOW_NSUB[0]}$, light "
              fr"$n_{{\rm sub}}={SHOW_NSUB[1]}$" "\n"
              fr"dotted: radial grazing angles ($n_{{\rm sub}}={SHOW_NSUB[0]}$)" "\n"
              "grey: reference uncertainty",
              ha="right", va="bottom", fontsize=7.5, color="0.3", linespacing=1.3)
    plt.setp(ax_c.get_xticklabels(), visible=False)
    ax_cr.set_ylim(1e-8, 30)
    ax_cr.set_yticks([1e-7, 1e-5, 1e-3, 1e-1])
    ax_cr.set_ylabel(r"$|\Delta|$ vs reference")
    ax_cr.set_xlabel(r"$\cos\theta_z$")

    # (d) summary of the geometry derivative ----------------------------------
    keep = np.abs(GEOM_CZ - CORE_CZ) > CORE_EXCL
    dscale = np.abs(dref[:, keep]).max()
    for seg in SCHEMES:
        p50, p95 = [], []
        for n in GEOM_NSUB:
            r = np.abs(np.asarray(geo[f"{seg}{n}"]["dP_dcz"]) - dref)[:, keep] / dscale
            p50.append(np.median(r))
            p95.append(np.percentile(r, 95))
        col, lab = style[seg]["color"], style[seg]["label"]
        ax_d.loglog(GEOM_NSUB, p95, "s--", color=col, ms=5, label=f"{lab}, 95th percentile")
        ax_d.loglog(GEOM_NSUB, p50, "o-", color=col, ms=5, label=f"{lab}, median")
    floor = np.percentile(ref_err[:, keep] / dscale, 95)
    ax_d.text(0.98, 0.97, f"reference uncertainty: {floor:.0e} (95th pct.)",
              transform=ax_d.transAxes, ha="right", va="top", color="0.4",
              fontsize=8.5)
    guide(ax_d, 1.0, 3.0e-3, 1, 64, label_offset=2.5, xm=36.0)
    mark_fig6(ax_d)
    ax_d.set_ylim(1e-8, 5e-2)
    ax_d.set_xlabel(r"$n_{\rm sub}$")
    ax_d.set_ylabel(r"$|\Delta\,\partial P_{\mu\mu}/\partial\cos\theta_z|\,/\,"
                    r"\max|\partial P_{\mu\mu}/\partial\cos\theta_z|$")
    ax_d.set_title(fr"(d) geometry derivative, {len(GEOM_E)} energies "
                   fr"{GEOM_E[0]:g}–{GEOM_E[-1]:g} GeV", loc="left", fontsize=11)
    ax_d.legend(fontsize=8.5, frameon=False, loc="lower left")

    base = os.path.join(OUTDIR, "mango_nsub_convergence")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    print("wrote", base + ".{pdf,png}")


def plot_fig6a(data):
    """Fig. 6(a) of the paper -- dmu_b/dln rho_core,bulk in reconstructed
    (E, cos zenith) space, nu channel -- repeated at several n_sub."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    sys.path.insert(0, HERE)
    import demo_tomography_fisher as T

    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})
    nb = T.N_E * T.N_CZ
    fish = {n: np.asarray(r["dmu_dcore"])[:nb].reshape(T.N_E, T.N_CZ)
            for n, r in _fisher_by_n(data).items()}
    ref = fish[FISHER_REF]
    scale = np.abs(ref).max()
    e_edges = np.logspace(np.log10(T.E_MIN), np.log10(T.E_MAX), T.N_E + 1)
    cz_edges = np.linspace(T.CZ_MIN, T.CZ_MAX, T.N_CZ + 1)
    e_c = np.sqrt(e_edges[:-1] * e_edges[1:])
    cz_c = 0.5 * (cz_edges[:-1] + cz_edges[1:])
    rows = [int(np.argmin(np.abs(e_c - e))) for e in FIG6A_SLICE_E]

    fig = plt.figure(figsize=(12.5, 8.4))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.15, 1], hspace=0.38)
    top = GridSpecFromSubplotSpec(1, 2, outer[0], width_ratios=[1, 1.35], wspace=0.3)
    ax_map = fig.add_subplot(top[0])
    sl = GridSpecFromSubplotSpec(2, 1, top[1], height_ratios=[1.5, 1], hspace=0.06)
    ax_s, ax_sr = fig.add_subplot(sl[0]), fig.add_subplot(sl[1])
    bot = GridSpecFromSubplotSpec(1, len(FIG6A_DIFF_NSUB) + 1, outer[1],
                                  width_ratios=[1] * len(FIG6A_DIFF_NSUB) + [0.05],
                                  wspace=0.12)

    # (a) the paper's panel at n_sub = 4, with the slice energies marked
    m = ax_map.pcolormesh(cz_edges, e_edges, fish[4], cmap="RdBu_r",
                          vmin=-scale, vmax=scale)
    fig.colorbar(m, ax=ax_map, label=r"$\partial\mu_b/\partial\ln\rho_{\rm core,bulk}$")
    for r, ls in zip(rows, ("--", ":")):
        ax_map.axhline(e_c[r], color="k", ls=ls, lw=1.1)
    ax_map.axvline(CORE_CZ, color="0.4", lw=0.8)
    ax_map.text(CORE_CZ + 0.015, 22, "core\nthreshold", color="0.3", fontsize=9, va="top")
    ax_map.set_yscale("log")
    ax_map.set_xlabel(r"reconstructed $\cos\theta_z$")
    ax_map.set_ylabel(r"reconstructed $E_\nu$ (GeV)")
    ax_map.set_title(r"(a) Fig. 6a ($n_{\rm sub}=4$, as in the paper)", loc="left", fontsize=11)

    # (b) slices at several n_sub, residual against n_sub = FISHER_REF
    cols = dict(zip(FIG6A_SLICE_NSUB, plt.cm.viridis(np.linspace(0.0, 0.85,
                                                                 len(FIG6A_SLICE_NSUB)))))
    for r, ls in zip(rows, ("--", ":")):
        ax_s.plot(cz_c, ref[r], color="0.6", lw=4, alpha=0.5,
                  label=fr"$n_{{\rm sub}}={FISHER_REF}$" if r == rows[0] else None)
        for n in FIG6A_SLICE_NSUB:
            ax_s.plot(cz_c, fish[n][r], color=cols[n], lw=1.2, ls=ls,
                      label=fr"$n_{{\rm sub}}={n}$" if r == rows[0] else None)
            ax_sr.semilogy(cz_c, np.abs(fish[n][r] - ref[r]) / scale + 1e-9,
                           color=cols[n], lw=1.1, ls=ls)
    for a in (ax_s, ax_sr):
        a.axvline(CORE_CZ, color="0.4", lw=0.8)
    ax_s.text(0.99, 0.04, "\n".join(fr"{ls}  $E_\nu={e_c[r]:.1f}$ GeV"
                                    for r, ls in zip(rows, ("dashed", "dotted"))),
              transform=ax_s.transAxes, ha="right", va="bottom", fontsize=9)
    ax_s.set_ylabel(r"$\partial\mu_b/\partial\ln\rho_{\rm core,bulk}$")
    ax_s.legend(fontsize=9, frameon=False, loc="center right", ncol=2,
                bbox_to_anchor=(1.0, 0.45))
    ax_s.set_title("(b) two energy rows of (a) at several $n_{\\rm sub}$", loc="left",
                   fontsize=11)
    plt.setp(ax_s.get_xticklabels(), visible=False)
    ax_sr.set_ylim(1e-6, 1e-1)
    ax_sr.set_yticks([1e-5, 1e-3, 1e-1])
    ax_sr.set_ylabel(r"$|\Delta|/\max|\partial\mu|$")
    ax_sr.set_xlabel(r"reconstructed $\cos\theta_z$")

    # (c) full-map differences against n_sub = FISHER_REF
    norm = SymLogNorm(linthresh=1e-4, vmin=-2e-2, vmax=2e-2, base=10)
    for k, n in enumerate(FIG6A_DIFF_NSUB):
        ax = fig.add_subplot(bot[k])
        diff = (fish[n] - ref) / scale
        m = ax.pcolormesh(cz_edges, e_edges, diff, cmap="PuOr_r", norm=norm)
        ax.axvline(CORE_CZ, color="0.4", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(r"reconstructed $\cos\theta_z$")
        if k == 0:
            ax.set_ylabel(r"reconstructed $E_\nu$ (GeV)")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(fr"({'cde'[k]}) $n_{{\rm sub}}={n}$ $-$ $n_{{\rm sub}}={FISHER_REF}$"
                     "\n" fr"max $|\Delta|$ = {np.abs(diff).max():.1e} of max",
                     loc="left", fontsize=10.5)
    cb = fig.colorbar(m, cax=fig.add_subplot(bot[-1]))
    cb.set_label(r"$\Delta\,\partial\mu_b\,/\,\max_b|\partial\mu_b|$")
    cb.set_ticks([-1e-2, -1e-3, -1e-4, 0, 1e-4, 1e-3, 1e-2])

    base = os.path.join(OUTDIR, "mango_nsub_fig6a")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    print("wrote", base + ".{pdf,png}")


PATH_STYLE = {
    "layered": dict(color="k", label="LayeredEarth (paper)"),
    "radial": dict(color="C3", label="PREM path, radial cut"),
    "chord": dict(color="C0", label="PREM path, chord cut"),
}


def plot_paths(data):
    """The tomography example on all three Earth paths of the demo
    (EARTH_PATH), against one common limit: the chord path at FISHER_REF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    sys.path.insert(0, HERE)
    import demo_tomography_fisher as T

    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})
    fish = {p: _fisher_by_n(data, p) for p in FISHER_PATHS}
    ref = fish["chord"][FISHER_REF]
    jref = np.asarray(ref["dmu_dcore"])
    scale = np.abs(jref).max()
    informative = np.abs(jref) > 1e-2 * scale

    def dev(path, n):
        return np.abs(np.asarray(fish[path][n]["dmu_dcore"]) - jref) / scale

    fig = plt.figure(figsize=(12.5, 8.8))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.1, 1], hspace=0.42)
    top = GridSpecFromSubplotSpec(1, 2, outer[0], wspace=0.3)
    ax_a, ax_b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    bot = GridSpecFromSubplotSpec(1, len(FISHER_PATHS) + 1, outer[1],
                                  width_ratios=[1] * len(FISHER_PATHS) + [0.05],
                                  wspace=0.12)

    # (a) Fig. 6a, whole map -------------------------------------------------
    for path in FISHER_PATHS:
        st = PATH_STYLE[path]
        ax_a.loglog(FISHER_NSUB, [dev(path, n).max() for n in FISHER_NSUB], "o-",
                    color=st["color"], ms=5, lw=1.2, label=st["label"])
        ax_a.loglog(FISHER_NSUB, [np.median(dev(path, n)[informative])
                                  for n in FISHER_NSUB], "D:",
                    color=st["color"], ms=4, lw=1.0)
    n2 = np.array([1.0, 32.0])
    ax_a.loglog(n2, 2e-3 * n2 ** -2, color="0.6", lw=0.8)
    ax_a.text(9, 2e-3 * 9 ** -2 * 0.3, r"$\propto n_{\rm sub}^{-2}$", color="0.45",
              fontsize=9)
    ax_a.axvline(4, color="0.55", ls=":", lw=1)
    lim = {p: dev(p, FISHER_REF).max() for p in ("layered", "radial")}
    ax_a.text(0.98, 0.97, "limits at $n_{\\rm sub}=%d$ vs chord:\n"
              "layered %.0e, radial %.0e" % (FISHER_REF, lim["layered"], lim["radial"]),
              transform=ax_a.transAxes, ha="right", va="top", fontsize=8.5,
              color="0.3")
    h = [Line2D([], [], color=PATH_STYLE[p]["color"], lw=1.2, label=PATH_STYLE[p]["label"])
         for p in FISHER_PATHS]
    h += [Line2D([], [], color="0.3", marker="o", ms=5, lw=1.2, label="maximum, all bins"),
          Line2D([], [], color="0.3", marker="D", ms=4, ls=":", lw=1.0,
                 label=r"median, bins $>1\%$ of max")]
    ax_a.legend(handles=h, fontsize=8.5, frameon=False, loc="lower left")
    ax_a.set_xlabel(r"$n_{\rm sub}$")
    ax_a.set_ylabel(r"$|\Delta\,\partial\mu_b/\partial\ln\rho_{\rm core,bulk}|"
                    r"\,/\,\max_b|\partial\mu_b|$")
    ax_a.set_title(r"(a) Fig. 6a, all bins, vs chord at $n_{\rm sub}=%d$" % FISHER_REF,
                   loc="left", fontsize=11)

    # (b) Fisher outputs: the worst of the five per path ---------------------
    keys = ["sigma_bulk_marg", "sigma_bulk_fixed", "sigma_ic_marg", "corr_ic_oc",
            "dsigma_dres"]
    for path in FISHER_PATHS:
        worst = [max(abs(fish[path][n][k] / ref[k] - 1.0) for k in keys)
                 for n in FISHER_NSUB]
        ax_b.loglog(FISHER_NSUB, worst, "o-", ms=5, lw=1.2,
                    color=PATH_STYLE[path]["color"], label=PATH_STYLE[path]["label"])
    ax_b.loglog(n2, 3e-2 * n2 ** -2, color="0.6", lw=0.8,
                label=r"$\propto n_{\rm sub}^{-2}$")
    ax_b.axvline(4, color="0.55", ls=":", lw=1)
    ax_b.text(0.98, 0.97, "largest of: $\\sigma$(bulk core), marginalized and fixed;\n"
              "$\\sigma$(inner core); corr(inner, outer core);\n"
              "$\\partial\\sigma/\\partial\\sigma_{\\cos\\theta_z}$",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=8.5, color="0.3")
    ax_b.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax_b.set_xlabel(r"$n_{\rm sub}$")
    ax_b.set_ylabel(r"relative deviation from chord, $n_{\rm sub}=%d$" % FISHER_REF)
    ax_b.set_title("(b) Fisher outputs of the tomography example", loc="left",
                   fontsize=11)

    # (c-e) where the n_sub = 4 error sits -----------------------------------
    e_edges = np.logspace(np.log10(T.E_MIN), np.log10(T.E_MAX), T.N_E + 1)
    cz_edges = np.linspace(T.CZ_MIN, T.CZ_MAX, T.N_CZ + 1)
    nb = T.N_E * T.N_CZ
    norm = SymLogNorm(linthresh=1e-5, vmin=-2e-3, vmax=2e-3, base=10)
    for k, path in enumerate(FISHER_PATHS):
        ax = fig.add_subplot(bot[k])
        diff = ((np.asarray(fish[path][4]["dmu_dcore"]) - jref)[:nb]
                .reshape(T.N_E, T.N_CZ) / scale)
        m = ax.pcolormesh(cz_edges, e_edges, diff, cmap="PuOr_r", norm=norm)
        ax.axvline(CORE_CZ, color="0.4", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(r"reconstructed $\cos\theta_z$")
        if k == 0:
            ax.set_ylabel(r"reconstructed $E_\nu$ (GeV)")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f"({'cde'[k]}) {PATH_STYLE[path]['label']}, $n_{{\\rm sub}}=4$\n"
                     fr"max $|\Delta|$ = {np.abs(diff).max():.1e} of max",
                     loc="left", fontsize=10.5)
    cb = fig.colorbar(m, cax=fig.add_subplot(bot[-1]))
    cb.set_label(r"$\Delta\,\partial\mu_b\,/\,\max_b|\partial\mu_b|$")
    cb.set_ticks([-1e-3, -1e-4, -1e-5, 0, 1e-5, 1e-4, 1e-3])

    base = os.path.join(OUTDIR, "mango_nsub_paths")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    print("wrote", base + ".{pdf,png}")


def plot_paper(data, lean=False):
    """The paper's convergence figure (Fig. 7): (a) part of Fig. 6a at several
    n_sub, (b) Fig. 6a and the Fisher outputs vs n_sub on the three Earth paths
    (``lean=True``: on the layered Earth of Fig. 6 only, against its own
    n_sub = FISHER_REF; written as mango_nsub_paper_lean),
    (c) dP/dcos(zenith) across the core threshold for both cuts, (d) its
    convergence.  Canvas and fonts match tomography_fisher (Fig. 6)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    sys.path.insert(0, HERE)
    import demo_tomography_fisher as T

    plt.rcParams.update({
        "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13,
        "axes.linewidth": 1.2, "lines.linewidth": 2.0, "mathtext.fontset": "cm",
    })
    fig = plt.figure(figsize=(11.0, 9.4))
    outer = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.34,
                     height_ratios=[1, 1])
    ga = GridSpecFromSubplotSpec(2, 1, outer[0, 0], height_ratios=[1.45, 1], hspace=0.07)
    gc = GridSpecFromSubplotSpec(2, 1, outer[1, 0], height_ratios=[1.45, 1], hspace=0.07)
    ax_a, ax_ar = fig.add_subplot(ga[0]), fig.add_subplot(ga[1])
    ax_c, ax_cr = fig.add_subplot(gc[0]), fig.add_subplot(gc[1])
    ax_b, ax_d = fig.add_subplot(outer[0, 1]), fig.add_subplot(outer[1, 1])
    used = dict(color="0.55", ls=":", lw=1.4)

    def n2_guide(ax, y1, lo=1.0, hi=32.0, label=True):
        n = np.array([lo, hi])
        ax.loglog(n, y1 * n ** -2, color="0.6", lw=1.0,
                  label=r"$\propto n_{\rm sub}^{-2}$" if label else None)

    # (a) two energy rows of Fig. 6a (LayeredEarth, as in the paper) -------
    nb = T.N_E * T.N_CZ
    lay = {n: np.asarray(r["dmu_dcore"])[:nb].reshape(T.N_E, T.N_CZ)
           for n, r in _fisher_by_n(data, "layered").items()}
    ref = lay[FISHER_REF]
    scale = np.abs(ref).max()
    e_edges = np.logspace(np.log10(T.E_MIN), np.log10(T.E_MAX), T.N_E + 1)
    cz_edges = np.linspace(T.CZ_MIN, T.CZ_MAX, T.N_CZ + 1)
    e_c = np.sqrt(e_edges[:-1] * e_edges[1:])
    cz_c = 0.5 * (cz_edges[:-1] + cz_edges[1:])
    rows = [int(np.argmin(np.abs(e_c - e))) for e in FIG6A_SLICE_E]
    nsl = [1, 2, 4]
    cols = dict(zip(nsl, ["#d95f02", "#7570b3", "#1b9e77"]))
    lss = ("-", "--")
    for r, ls in zip(rows, lss):
        ax_a.plot(cz_c, ref[r] / scale, color="0.65", lw=5, alpha=0.6, ls=ls)
        for n in nsl:
            ax_a.plot(cz_c, lay[n][r] / scale, color=cols[n], lw=1.6, ls=ls)
            ax_ar.semilogy(cz_c, np.abs(lay[n][r] - ref[r]) / scale + 1e-12,
                           color=cols[n], lw=1.4, ls=ls)
    for a in (ax_a, ax_ar):
        a.axvline(CORE_CZ, color="0.4", lw=0.9)
        a.set_xlim(cz_edges[0], -0.2)
    h = [Line2D([], [], color="0.65", lw=5, alpha=0.6,
                label=fr"$n_{{\rm sub}}={FISHER_REF}$")]
    h += [Line2D([], [], color=cols[n], lw=1.6, label=fr"$n_{{\rm sub}}={n}$") for n in nsl]
    h += [Line2D([], [], color="k", lw=1.4, ls=ls, label=fr"$E_\nu={e_c[r]:.1f}$ GeV")
          for r, ls in zip(rows, lss)]
    ax_a.legend(handles=h, loc="lower right", frameon=False, fontsize=12,
                handlelength=1.8, labelspacing=0.3, borderaxespad=0.2)
    ax_a.set_ylabel(r"$\partial\mu_b/\partial\ln\rho_{\rm core}$" "\n(norm.)")
    ax_a.set_title("(a) rows of Fig. 6(a) vs $n_{\\rm sub}$", loc="left")
    ax_ar.text(CORE_CZ + 0.012, 0.95, "core threshold", transform=ax_ar.get_xaxis_transform(),
               color="0.35", fontsize=12, va="top")
    plt.setp(ax_a.get_xticklabels(), visible=False)
    ax_ar.set_ylim(1e-6, 0.3)
    ax_ar.set_yticks([1e-5, 1e-3, 1e-1])
    ax_ar.set_ylabel(r"$|\Delta|$")
    ax_ar.set_xlabel(r"reconstructed $\cos\theta_z$")

    keys = ["sigma_bulk_marg", "sigma_bulk_fixed", "sigma_ic_marg", "corr_ic_oc",
            "dsigma_dres"]
    if lean:
        # (b) the layered Earth of Fig. 6 only, against its own n_sub = FISHER_REF
        lf = _fisher_by_n(data, "layered")
        lref = lf[FISHER_REF]
        jref = np.asarray(lref["dmu_dcore"])
        jscale = np.abs(jref).max()
        informative = np.abs(jref) > 1e-2 * jscale
        dev = {n: np.abs(np.asarray(lf[n]["dmu_dcore"]) - jref) / jscale
               for n in FISHER_NSUB}
        ax_b.loglog(FISHER_NSUB, [dev[n].max() for n in FISHER_NSUB], "o-", color="k",
                    ms=6, lw=1.6, label="Fig. 6(a), max over bins")
        ax_b.loglog(FISHER_NSUB, [np.median(dev[n][informative]) for n in FISHER_NSUB],
                    "D:", color="0.45", ms=5, lw=1.6,
                    label=r"Fig. 6(a), median, bins $>1\%$ of max")
        ax_b.loglog(FISHER_NSUB, [max(abs(lf[n][k] / lref[k] - 1.0) for k in keys)
                                  for n in FISHER_NSUB], "s--", color="#7b3294", ms=6,
                    lw=1.6, mfc="none", label="Fisher outputs, largest")
        n2_guide(ax_b, 3e-2)
        ax_b.axvline(4, **used)
        ax_b.text(4.3, 0.93, "Fig. 6", transform=ax_b.get_xaxis_transform(),
                  color="0.4", fontsize=12, va="top")
        ax_b.legend(loc="lower left", frameon=False, fontsize=12)
        ax_b.set_ylim(1e-7, 0.2)
        ax_b.set_xlabel(r"$n_{\rm sub}$")
        ax_b.set_ylabel(fr"deviation from $n_{{\rm sub}}={FISHER_REF}$")
        ax_b.set_title("(b) tomography example vs $n_{\\rm sub}$", loc="left")
    else:
        # (b) Fig. 6a and the Fisher outputs vs n_sub, three Earth paths ---------
        fish = {p: _fisher_by_n(data, p) for p in FISHER_PATHS}
        cref = fish["chord"][FISHER_REF]
        jref = np.asarray(cref["dmu_dcore"])
        jscale = np.abs(jref).max()
        lab = {"layered": "layered Earth (Fig. 6)", "radial": "PREM, radial cut",
               "chord": "PREM, chord cut"}
        for path in FISHER_PATHS:
            c = PATH_STYLE[path]["color"]
            map_err = [np.abs(np.asarray(fish[path][n]["dmu_dcore"]) - jref).max() / jscale
                       for n in FISHER_NSUB]
            worst = [max(abs(fish[path][n][k] / cref[k] - 1.0) for k in keys)
                     for n in FISHER_NSUB]
            ax_b.loglog(FISHER_NSUB, map_err, "o-", color=c, ms=6, lw=1.6)
            ax_b.loglog(FISHER_NSUB, worst, "s--", color=c, ms=5, lw=1.4, mfc="none")
        n2_guide(ax_b, 3e-2, label=False)
        ax_b.axvline(4, **used)
        ax_b.text(4.3, 0.93, "Fig. 6", transform=ax_b.get_xaxis_transform(),
                  color="0.4", fontsize=12, va="top")
        h = [Line2D([], [], color=PATH_STYLE[p]["color"], lw=1.6, label=lab[p])
             for p in FISHER_PATHS]
        h += [Line2D([], [], color="0.3", marker="o", ms=6, lw=1.6,
                     label=r"Fig. 6(a), max over bins"),
              Line2D([], [], color="0.3", marker="s", ms=5, mfc="none", ls="--", lw=1.4,
                     label="Fisher outputs, worst"),
              Line2D([], [], color="0.6", lw=1.0, label=r"$\propto n_{\rm sub}^{-2}$")]
        ax_b.legend(handles=h, loc="lower left", frameon=False, fontsize=12)
        ax_b.set_ylim(1e-9, 0.3)
        ax_b.set_xlabel(r"$n_{\rm sub}$")
        ax_b.set_ylabel(fr"deviation from chord, $n_{{\rm sub}}={FISHER_REF}$")
        ax_b.set_title("(b) tomography example vs $n_{\\rm sub}$", loc="left")

    # (c) dP/dcos(zenith) across the core threshold -------------------------
    geo = data["geometry"]
    dref = np.asarray(geo[f"chord{GEOM_REF}"]["dP_dcz"])
    ie = int(np.argmin(np.abs(GEOM_E - GEOM_SHOW_E)))
    seg_col = {"radial": PATH_STYLE["radial"]["color"],
               "chord": PATH_STYLE["chord"]["color"]}
    seg_lab = {"radial": "radial cut", "chord": "chord cut"}
    n0, n1 = SHOW_NSUB
    ax_c.plot(GEOM_CZ, dref[ie], color="0.65", lw=5, alpha=0.6,
              label=fr"chord, $n_{{\rm sub}}={GEOM_REF}$")
    for seg in SCHEMES:
        d0 = np.asarray(geo[f"{seg}{n0}"]["dP_dcz"])[ie]
        ax_c.plot(GEOM_CZ, d0, color=seg_col[seg], lw=1.3,
                  label=fr"{seg_lab[seg]}, $n_{{\rm sub}}={n0}$")
        for n, alpha in ((n0, 1.0), (n1, 0.4)):
            d = np.asarray(geo[f"{seg}{n}"]["dP_dcz"])[ie]
            ax_cr.semilogy(GEOM_CZ, np.abs(d - dref[ie]) + 1e-12, color=seg_col[seg],
                           alpha=alpha, lw=1.0)
    for a in (ax_c, ax_cr):
        a.axvspan(CORE_CZ - CORE_EXCL, CORE_CZ + CORE_EXCL, color="0.5", alpha=0.18, lw=0)
        a.set_xlim(GEOM_CZ[0], GEOM_CZ[-1])
    ax_c.set_ylim(-27, 48)
    ax_c.set_ylabel(r"$\partial P_{\mu\mu}/\partial\cos\theta_z$")
    ax_c.set_title(fr"(c) zenith derivative, $E_\nu={GEOM_E[ie]:g}$ GeV", loc="left")
    ax_c.legend(loc="upper left", frameon=False, fontsize=12)
    ax_c.text(CORE_CZ + 0.004, -24, "core threshold", color="0.35", fontsize=12,
              va="bottom")
    plt.setp(ax_c.get_xticklabels(), visible=False)
    ax_cr.set_ylim(1e-7, 300)
    ax_cr.set_yticks([1e-6, 1e-3, 1])
    ax_cr.set_ylabel(r"$|\Delta|$")
    ax_cr.set_xlabel(r"$\cos\theta_z$")
    ax_cr.text(0.99, 0.04, fr"dark $n_{{\rm sub}}={n0}$, light ${n1}$",
               transform=ax_cr.transAxes, ha="right", va="bottom", fontsize=12,
               color="0.3")

    # (d) convergence of (c) over 2-20 GeV ----------------------------------
    keep = np.abs(GEOM_CZ - CORE_CZ) > CORE_EXCL
    dscale = np.abs(dref[:, keep]).max()
    for seg in SCHEMES:
        p50, p95 = [], []
        for n in GEOM_NSUB:
            r = np.abs(np.asarray(geo[f"{seg}{n}"]["dP_dcz"]) - dref)[:, keep] / dscale
            p50.append(np.median(r))
            p95.append(np.percentile(r, 95))
        ax_d.loglog(GEOM_NSUB, p95, "s--", color=seg_col[seg], ms=5, lw=1.4, mfc="none")
        ax_d.loglog(GEOM_NSUB, p50, "o-", color=seg_col[seg], ms=6, lw=1.6)
    n2_guide(ax_d, 3e-3, hi=64.0, label=False)
    ax_d.text(20, 3e-3 * 20 ** -2 * 3.0, r"$\propto n_{\rm sub}^{-2}$", color="0.45",
              fontsize=13)
    ax_d.axvline(4, **used)
    h = [Line2D([], [], color=seg_col[s], lw=1.6, label=seg_lab[s]) for s in SCHEMES]
    h += [Line2D([], [], color="0.3", marker="s", ms=5, mfc="none", ls="--", lw=1.4,
                 label="95th percentile"),
          Line2D([], [], color="0.3", marker="o", ms=6, lw=1.6, label="median")]
    ax_d.legend(handles=h, loc="lower left", frameon=False, fontsize=12)
    ax_d.set_ylim(1e-8, 5e-2)
    ax_d.set_xlabel(r"$n_{\rm sub}$")
    ax_d.set_ylabel(r"$|\Delta\,\partial P_{\mu\mu}/\partial\cos\theta_z|$ / max")
    ax_d.set_title(fr"(d) zenith derivative, {GEOM_E[0]:g}–{GEOM_E[-1]:g} GeV",
                   loc="left")

    base = os.path.join(OUTDIR, "mango_nsub_paper" + ("_lean" if lean else ""))
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    print("wrote", base + ".{pdf,png}")


def plot_paper_single(data, show_radial=True):
    """Single-panel, single-column alternative for the paper: the worst-case
    deviation of every derivative the paper uses, against n_sub.  One message:
    the derivatives converge, and n_sub = 4 is enough; the radial-cut line is
    the one contrast (pointwise zenith derivatives need the chord cut)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9, "axes.linewidth": 0.8,
        "mathtext.fontset": "cm",
    })
    keys = ["sigma_bulk_marg", "sigma_bulk_fixed", "sigma_ic_marg", "corr_ic_oc",
            "dsigma_dres"]
    lf = _fisher_by_n(data, "layered")
    lref = lf[FISHER_REF]
    jref = np.asarray(lref["dmu_dcore"])
    map_err = [np.abs(np.asarray(lf[n]["dmu_dcore"]) - jref).max() / np.abs(jref).max()
               for n in FISHER_NSUB]
    fish_err = [max(abs(lf[n][k] / lref[k] - 1.0) for k in keys) for n in FISHER_NSUB]
    geo = data["geometry"]
    dref = np.asarray(geo[f"chord{GEOM_REF}"]["dP_dcz"])
    keep = np.abs(GEOM_CZ - CORE_CZ) > CORE_EXCL
    dscale = np.abs(dref[:, keep]).max()

    def geo_err(seg):
        return [np.abs(np.asarray(geo[f"{seg}{n}"]["dP_dcz"]) - dref)[:, keep].max()
                / dscale for n in GEOM_NSUB]

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.axvline(4, color="0.6", ls=":", lw=1.0)
    n = np.array([1.0, 64.0])
    ax.loglog(n, 1e-2 * n ** -2, color="0.7", lw=0.9, label=r"$\propto n_{\rm sub}^{-2}$")
    ax.loglog(FISHER_NSUB, map_err, "o-", color="k", ms=4, lw=1.3,
              label=r"$\partial\mu_b/\partial\ln\rho_{\rm core}$, Fig. 6(a)")
    ax.loglog(FISHER_NSUB, fish_err, "s--", color="#7b3294", ms=4, lw=1.3, mfc="none",
              label="Fisher outputs, Fig. 6(c,d)")
    ax.loglog(GEOM_NSUB, geo_err("chord"), "D-", color="C0", ms=4, lw=1.3,
              label=r"$\partial P_{\mu\mu}/\partial\cos\theta_z$ near core threshold")
    if show_radial:
        ax.loglog(GEOM_NSUB, geo_err("radial"), "D:", color="C3", ms=3.5, lw=1.1,
                  mfc="none", label=r"same, with the radial sub-shell cut")
    ax.text(4.25, 0.97, r"$n_{\rm sub}=4$" "\n(Fig. 6)", transform=ax.get_xaxis_transform(),
            color="0.4", fontsize=8.5, va="top")
    ax.set_xlim(0.8, 80)
    ax.set_ylim(5e-7, 2.0)
    ax.set_xlabel(r"$n_{\rm sub}$")
    ax.set_ylabel("worst-case relative deviation")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), frameon=False,
              handlelength=2.2, ncol=1)
    base = os.path.join(OUTDIR, "mango_nsub_single" + ("" if show_radial else "_noradial"))
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=250, bbox_inches="tight")
    print("wrote", base + ".{pdf,png}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--fisher-worker":
        _fisher_worker(int(sys.argv[2]), *sys.argv[3:4])
        sys.exit(0)
    data = _load_cache()
    if "--plot" not in sys.argv:
        print("Fisher outputs vs n_sub (LayeredEarth, demo_tomography_fisher)")
        compute_fisher(data)
        print("dP/dcz vs n_sub (PREM path, radial and chord segmentation)")
        compute_geometry(data)
    plot(data)
    plot_fig6a(data)
    plot_paths(data)
    plot_paper(data)
    plot_paper(data, lean=True)
    plot_paper_single(data)
