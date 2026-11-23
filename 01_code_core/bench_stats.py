"""
bench_stats.py — تحلیل آماری benchmark مشترک (بدون وابستگی به env/SB3؛ مستقل قابل‌تست)

ورودی: DataFrame اپیزودها با ستون‌های
  iw, scenario, type, ctrl, seed, crashed, rmse, u_rms, du, [cbf_*] ...
نام ctrl: 'nmpc' | 'res:<stem>' | 'sac_plain_sK' | 'sac_hocbf_sK' | 'sac_hocbf_oracle_sK' | 'mpc'
خانواده (fam): residual | nmpc | mpc | sac_plain | sac_hocbf | sac_hocbf_oracle

اصول:
  * تحلیل اصلی روی «همه‌ی seedها» است. تحلیل بدون early-failure فقط تکمیلی است (tag="_excl_early").
  * برای خانواده‌های چندمدلی (SAC با ۵ seed آموزشی) دو منبع تغییر داریم: مدل‌های آموزشی و seedهای اپیزود.
    CI بقا با bootstrap دوسطحی (نمونه‌گیری مدل‌ها، بعد اپیزودها) محاسبه می‌شود؛ Wilson روی داده‌ی pooled
    عمداً برای خانواده‌ها استفاده نمی‌شود چون خوشه‌ای بودن داده‌ها را نادیده می‌گیرد.
  * مقایسه‌ی زوجی: اپیزودهای نمونه‌گیری‌شده (همان شرایط اولیه/نویز) برای هر دو طرف یکسان‌اند.
"""
import math, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", message="Mean of empty slice")

SENSOR = {"bias_phi", "noise_phi", "drift_phi", "stuck_phi", "dropout_phi", "intermittent_phi", "multifault"}
SAC_FAMS = ("sac_plain", "sac_hocbf", "sac_hocbf_oracle")
COMPARISONS = [("residual", "nmpc"), ("sac_plain", "nmpc"), ("sac_hocbf", "nmpc"),
               ("residual", "sac_plain"), ("residual", "sac_hocbf"), ("sac_hocbf", "sac_plain"),
               ("sac_hocbf_oracle", "sac_hocbf"), ("mpc", "nmpc"), ("residual", "mpc")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mcnemar_exact(only_a, only_b):
    """دوطرفه‌ی دقیق؛ NaN اگر هیچ جفت ناهمخوانی نباشد."""
    n = only_a + only_b
    if n == 0:
        return np.nan
    k = min(only_a, only_b)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(p, 1.0))


def fam_of(ctrl):
    import re
    return "residual" if ctrl.startswith("res:") else re.sub(r"_s\d+$", "", ctrl)


def grp_of(ftype):
    return "healthy" if ftype == "none" else "sensor" if ftype in SENSOR else "param" if ftype == "param_fault" else "actuator"


def _mats(d):
    """d: اپیزودهای یک (iw, scenario). → {fam: (alive[M,E], rmse[M,E], names)} روی seedهای مشترکِ همه‌ی ctrlها."""
    P_c = d.pivot_table(index="ctrl", columns="seed", values="crashed", aggfunc="first")
    P_r = d.pivot_table(index="ctrl", columns="seed", values="rmse", aggfunc="first")
    P_c = P_c.dropna(axis=1); P_r = P_r[P_c.columns]
    out = {}
    for f in sorted({fam_of(c) for c in P_c.index}):
        names = [c for c in P_c.index if fam_of(c) == f]
        al = ~P_c.loc[names].to_numpy().astype(bool)
        rm = P_r.loc[names].to_numpy().astype(float); rm = np.where(al, rm, np.nan)
        out[f] = (al, rm, names)
    return out, list(P_c.columns)


def _boot_surv(al, ep, mi):
    return al[mi[:, :, None], ep[:, None, :]].mean(axis=(1, 2))


def paired_and_family(d, B, rng):
    """برگشت: (family_rows, paired_surv_rows, paired_rmse_rows) برای یک (iw, scenario)."""
    mats, seeds = _mats(d)
    E = len(seeds)
    if E == 0:
        return [], [], []
    ep = rng.integers(0, E, (B, E))                                   # مشترک بین همه‌ی خانواده‌ها ⇒ زوجی
    mi = {f: rng.integers(0, m[0].shape[0], (B, m[0].shape[0])) for f, m in mats.items()}
    bs = {f: _boot_surv(m[0], ep, mi[f]) for f, m in mats.items()}
    fam_rows, ps_rows, pr_rows = [], [], []
    for f, (al, rm, names) in mats.items():
        per_model = al.mean(axis=1)
        lo, hi = np.percentile(bs[f], [2.5, 97.5])
        fam_rows.append(dict(fam=f, n_models=len(names), n_eps=E, surv=float(per_model.mean()),
                             surv_std_models=float(per_model.std(ddof=1)) if len(names) > 1 else np.nan,
                             surv_min_model=float(per_model.min()), surv_max_model=float(per_model.max()),
                             surv_ci_lo=float(lo), surv_ci_hi=float(hi),
                             rmse_alive=float(np.nanmean(rm)) if np.isfinite(rm).any() else np.nan))
    for A, Bf in COMPARISONS:
        if A not in mats or Bf not in mats:
            continue
        aA, rA, _ = mats[A]; aB, rB, _ = mats[Bf]
        dobs = float(aA.mean() - aB.mean()); dd = bs[A] - bs[Bf]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        row = dict(A=A, B=Bf, n_eps=E, dsurv=dobs, ci_lo=float(lo), ci_hi=float(hi),
                   frac_boot_A_better=float((dd > 0).mean()), p_mcnemar=np.nan)
        if aA.shape[0] == 1 and aB.shape[0] == 1:                     # تک‌مدل در برابر تک‌مدل: آزمون دقیق
            row["p_mcnemar"] = mcnemar_exact(int((aA[0] & ~aB[0]).sum()), int((~aA[0] & aB[0]).sum()))
        else:                                                         # چندمدلی: چند مدل A از میانگین B بهتر است
            row["models_A_better"] = int((aA.mean(axis=1) > aB.mean()).sum()) if aA.shape[0] > 1 else np.nan
        ps_rows.append(row)
        vA = np.nanmean(np.where(aA, rA, np.nan), axis=0) if np.isfinite(rA).any() else np.full(E, np.nan)
        vB = np.nanmean(np.where(aB, rB, np.nan), axis=0) if np.isfinite(rB).any() else np.full(E, np.nan)
        diff = vA - vB; valid = np.isfinite(diff)
        if valid.sum() >= 2:
            s = np.where(valid, diff, 0.0)[ep].sum(1); c = valid[ep].sum(1)
            m = s / np.maximum(c, 1); m = m[c > 0]
            lo, hi = np.percentile(m, [2.5, 97.5])
            pr_rows.append(dict(A=A, B=Bf, n_both_alive=int(valid.sum()), d_rmse=float(diff[valid].mean()),
                                ci_lo=float(lo), ci_hi=float(hi),
                                d_rmse_pct=float(diff[valid].mean() / np.nanmean(vB[valid]) * 100)))
    return fam_rows, ps_rows, pr_rows


def analyze(D, out, tag="", B=2000, seed=0, verbose=True):
    """D شامل ستون fam (یا از ctrl ساخته می‌شود). فایل‌ها: <out><tag>_{summary,family,by_group,paired_survival,paired_rmse}.csv"""
    D = D.copy()
    if "fam" not in D:
        D["fam"] = D.ctrl.map(fam_of)
    rng = np.random.default_rng(seed)

    def agg(x):
        k = int((~x.crashed).sum()); lo, hi = wilson(k, len(x)); al = x[~x.crashed]
        o = dict(n=len(x), surv=k / max(len(x), 1), ci_lo=lo, ci_hi=hi, rmse_alive=al.rmse.mean(),
                 peak_phi=x.peak_phi.mean(), u_rms=x.u_rms.mean(), u2_sum=x.u2_sum.mean(), du=x.du.mean(),
                 sat=x.sat.mean(), ms=x.ms.mean())
        for c in ("inj_ok", "freq_ok"):
            if c in x:
                o[c] = x[c].mean()
        return pd.Series(o)

    S = D.groupby(["iw", "scenario", "ctrl"]).apply(agg, include_groups=False)
    S.to_csv(f"{out}{tag}_summary.csv")
    FR, PS, PR = [], [], []
    ftype = D.drop_duplicates("scenario").set_index("scenario")["type"].to_dict()
    for (iw, sc), d in D.groupby(["iw", "scenario"]):
        f, ps, pr = paired_and_family(d, B, rng)
        for r in f: FR.append(dict(iw=iw, scenario=sc, **r))
        for r in ps: PS.append(dict(iw=iw, scenario=sc, **r))
        for r in pr: PR.append(dict(iw=iw, scenario=sc, **r))
    F, P1, P2 = pd.DataFrame(FR), pd.DataFrame(PS), pd.DataFrame(PR)
    F["group"] = F.scenario.map(lambda s: grp_of(ftype[s]))
    F.to_csv(f"{out}{tag}_family.csv", index=False)
    P1.to_csv(f"{out}{tag}_paired_survival.csv", index=False)
    P2.to_csv(f"{out}{tag}_paired_rmse.csv", index=False)
    GS = F.groupby(["iw", "group", "fam"])[["surv", "surv_std_models", "rmse_alive"]].mean()
    GS.to_csv(f"{out}{tag}_by_group.csv")
    if verbose:
        pd.set_option("display.width", 250); pd.set_option("display.max_rows", 600)
        print(f"\n===== [{tag or 'ALL SEEDS (گزارش اصلی)'}] میانگین هر گروه (وزن برابر هر سناریو) =====\n", GS.round(3).to_string())
        T = F.pivot_table(index=["iw", "scenario"], columns="fam", values="surv")
        print(f"\n===== [{tag or 'ALL SEEDS'}] survival به‌تفکیک خانواده =====\n", T.round(3).to_string())
    return dict(summary=S, family=F, paired_surv=P1, paired_rmse=P2)


HOCBF_COLS = ["cbf_filter_rate", "cbf_slack_rate", "cbf_qp_fail_rate", "cbf_cond_viol_meas", "cbf_cond_viol_true",
              "cbf_psi1_viol_true", "cbf_phiddot_rmse", "cbf_phiddot_bias",
               "cbf_qp_attempt_rate", "cbf_qp_success_rate", "cbf_fallback_rate", "cbf_bypass_rate",
               "cbf_slack_mean_on_solve", "cbf_slack_max", "cbf_cond_margin_meas_mean",
               "cbf_cond_margin_true_mean", "cbf_cond_margin_true_min"]


def hocbf_diag(D, out):
    """رفتار فیلتر جدا از نرخ بقا: slack، شکست QP، نقض شرط CBF با state اندازه‌گیری‌شده/واقعی، و خطای تقریب φ̈."""
    cols = [c for c in HOCBF_COLS if c in D]
    H = D[D.ctrl.str.startswith("sac_hocbf") & D[cols].notna().any(axis=1)] if cols else D.iloc[0:0]
    if H.empty:
        return None
    G = H.groupby(["iw", "scenario", "fam"])[cols].mean()
    G.to_csv(f"{out}_hocbf_diag.csv")
    return G
