"""Per-scenario locomotion metrics from recorded robot states. Numpy only, so the same code scores the MJX/Warp rollouts
(rollout.py) and the CPU MuJoCo runs of the champ_mujoco simulation (champ_transfer), whatever produced the records.

`compute` takes a dict of arrays shaped (robots, T, ...), sampled every `dt` seconds, robots grouped per scenario:
    done      (robots, T)       1 once the robot has fallen (upvector z < 0, the environment's termination)
    linvel    (robots, T, 3)    body linear velocity in body (imu) axes
    yaw_rate  (robots, T)       world z angular velocity
    up_z      (robots, T)       z of the body up axis in world axes
    tau, qd   (robots, T, 12)   actuator torque, joint velocity
    pos       (robots, T, 3)    base position (only x, y are used: path length)
    contact   (robots, T, 4)    foot on the floor
    foot_z    (robots, T, 4)    foot site height
    foot_vxy  (robots, T, 4)    foot horizontal speed
  optional
    action    (robots, T, 12)   policy output (jerk column; blank without it)
    terms, reward               RL reward terms (with `term_keys`) and total reward (reward table; skipped without them)
and returns (rows, text): one dict of numbers per scenario (the csv columns) and the printed tables.
"""
import numpy as np

# (vx m/s, vy m/s, wz rad/s): the fixed joystick commands every evaluation runs.
SCENARIOS = {
    "stand":        (0.0, 0.0, 0.0),
    "forward":      (0.3, 0.0, 0.0),
    "forward_max":  (0.50, 0.0, 0.0),
    "backward":     (-0.15, 0.0, 0.0),
    "left":         (0.0, 0.08, 0.0),
    "turn":         (0.0, 0.0, 0.8),
    "arc":          (0.12, 0.0, 0.5),
    "diagonal":     (0.12, 0.06, 0.0),
}


def compute(r, names, scenarios, per, dt, settle, total_mass, torque_limit, max_joint_velocity, term_keys=()):
    lines = []
    out = lines.append
    T = r["done"].shape[1]
    alive = np.cumprod(1.0 - r["done"], axis=1) > 0                            # still upright after step t
    t0 = int(round(settle / dt))
    power = np.abs(r["tau"] * r["qd"]).sum(-1)                                 # W
    sat = np.abs(r["tau"]) >= 0.98 * torque_limit
    fast = np.abs(r["qd"]) > max_joint_velocity
    tilt = np.degrees(np.arccos(np.clip(r["up_z"], -1, 1)))
    jerk = (np.abs(np.diff(r["action"], axis=1, prepend=r["action"][:, :1])).mean(-1) if "action" in r
            else np.full(r["done"].shape, np.nan))
    step_dist = np.linalg.norm(np.diff(r["pos"][..., :2], axis=1, prepend=r["pos"][:, :1, :2]), axis=-1)

    rows = {name: dict(scenario=name) for name in names}
    hdr = (f"{'scenario':<12}{'command':>18} {'alive':>6} {'vx':>14} {'vy':>14} {'wz':>14} {'CoT':>6} {'P[W]':>6} "
           f"{'tilt':>10} {'sat':>6} {'>vmax':>6} {'jerk':>5}")
    out(hdr + "\n" + "-" * len(hdr))
    for i, name in enumerate(names):
        sl = slice(i * per, (i + 1) * per)
        ok = alive[sl, -1]
        cmd = scenarios[name]
        survivors = int(ok.sum())
        rec = rows[name]
        rec.update(cmd_vx=cmd[0], cmd_vy=cmd[1], cmd_wz=cmd[2], alive=survivors, envs=per)
        row = f"{name:<12}{str(tuple(cmd)):>18} {survivors:>3}/{per:<2}"
        if survivors == 0:
            out(row + f"   all fell, after {dt * alive[sl].sum(1).mean():.1f} s on average")
            continue
        lv = r["linvel"][sl][ok][:, t0:]
        yr = r["yaw_rate"][sl][ok][:, t0:]
        stats = []
        for key, sig, c in (("vx", lv[..., 0], cmd[0]), ("vy", lv[..., 1], cmd[1]), ("wz", yr, cmd[2])):
            rms = np.sqrt(((sig - c) ** 2).mean())
            rec[key], rec[key + "_rms_err"] = sig.mean(), rms
            stats.append(f"{sig.mean():+.3f}±{rms:.3f}")
        dist = step_dist[sl][ok][:, t0:].sum(1)
        energy = (power[sl][ok][:, t0:] * dt).sum(1)
        cot = energy / (total_mass * np.maximum(dist, 1e-9))
        moving = np.linalg.norm(cmd[:2]) > 0.02
        cot_s = f"{np.median(cot):6.2f}" if moving else f"{'-':>6}"
        tl = tilt[sl][ok][:, t0:]
        jk = jerk[sl][ok][:, t0:].mean()
        rec.update(cot=float(np.median(cot)) if moving else np.nan, power_w=power[sl][ok][:, t0:].mean(),
                   tilt_mean_deg=tl.mean(), tilt_max_deg=tl.max(), saturation_pct=100 * sat[sl][ok][:, t0:].mean(),
                   over_vmax_pct=100 * fast[sl][ok][:, t0:].mean(), jerk=jk)
        jerk_s = f"{jk:5.2f}" if np.isfinite(jk) else f"{'-':>5}"
        row += (f" {stats[0]:>14} {stats[1]:>14} {stats[2]:>14} {cot_s} {power[sl][ok][:, t0:].mean():6.3f} "
                f"{tl.mean():4.1f}/{tl.max():4.1f}° {100 * sat[sl][ok][:, t0:].mean():5.1f}% {100 * fast[sl][ok][:, t0:].mean():5.1f}% "
                f"{jerk_s}")
        out(row)
    out("\ngait, averaged over feet (cadence = touchdowns/s/foot, clearance = mean swing peak above its own liftoff/touchdown height):")
    out(f"{'scenario':<12}{'cadence':>9}{'duty':>7}{'clearance':>9}{'stride':>9}{'4 feet off':>16}{'v stance':>11}{'v air':>10}   (feet speed in mm/s)")
    for i, name in enumerate(names):
        sl = slice(i * per, (i + 1) * per)
        ok = alive[sl, -1]
        if not ok.any():
            continue
        c = r["contact"][sl][ok][:, t0:].astype(bool)                         # (robots, T', 4)
        z = r["foot_z"][sl][ok][:, t0:]
        td = (c[:, 1:] & ~c[:, :-1]).sum((1, 2)) / 4 / (c.shape[1] * dt)     # touchdowns / s / foot
        peaks, vair = [], []
        fv = r["foot_vxy"][sl][ok][:, t0:]
        for e in range(c.shape[0]):
            for f in range(4):
                air = ~c[e, :, f]
                edges = np.flatnonzero(np.diff(np.r_[0, air.astype(int), 0]))
                for a, b in zip(edges[::2], edges[1::2]):
                    # clearance of this swing: above the HIGHER of its liftoff and touchdown heights, so terrain relief
                    # does not count as lift; swings cut off by the start or end of the window are skipped
                    if a > 0 and b < len(air):
                        peaks.append(z[e, a:b, f].max() - max(z[e, a - 1, f], z[e, b, f]))
                vair.append(fv[e, air, f].mean() if air.any() else np.nan)
        dist = step_dist[sl][ok][:, t0:].sum(1) / np.maximum(td * 4 * c.shape[1] * dt, 1e-9)
        swing = f"{1e3 * np.mean(peaks):6.1f} mm" if peaks else f"{'-':>9}"
        stride_ok = np.linalg.norm(scenarios[name][:2]) > 0.02 and td.mean() > 0.1
        stride = f"{1e3 * np.median(dist):6.1f} mm" if stride_ok else f"{'-':>9}"
        rows[name].update(
            cadence_hz=td.mean(), duty_pct=100 * c.mean(), clearance_mm=1e3 * np.mean(peaks) if peaks else np.nan,
            stride_mm=1e3 * np.median(dist) if stride_ok else np.nan, four_feet_off_pct=100 * (~c).all(2).mean(),
            v_stance_mm_s=1e3 * fv[c].mean() if c.any() else np.nan,
            v_air_mm_s=1e3 * np.nanmean(vair) if np.isfinite(vair).any() else np.nan)
        out(f"{name:<12}{td.mean():7.2f}Hz{100 * c.mean():6.0f}%{swing:>9}{stride:>9}{100 * (~c).all(2).mean():15.1f}%"
            f"{1e3 * fv[c].mean() if c.any() else 0:11.1f}{1e3 * np.nanmean(vair) if np.isfinite(vair).any() else 0:10.1f}")
    if "terms" in r and term_keys:
        out("\nreward terms per control step, already scaled by their weights (before the *dt in the reward); "
            "negative = penalty:")
        keys = [k.split("/", 1)[1] for k in term_keys]
        out(f"{'scenario':<12}" + "".join(f"{k[:11]:>12}" for k in keys) + f"{'total*dt':>10}")
        for i, name in enumerate(names):
            sl = slice(i * per, (i + 1) * per)
            ok = alive[sl, -1]
            if ok.any():
                tm = r["terms"][sl][ok][:, t0:].mean((0, 1))
                rows[name].update({f"term_{k}": v for k, v in zip(keys, tm)}, term_total_dt=r["reward"][sl][ok][:, t0:].mean())
                out(f"{name:<12}" + "".join(f"{v:12.4f}" for v in tm) + f"{r['reward'][sl][ok][:, t0:].mean():10.4f}")
    out("\nvelocity columns are mean±RMS error against the command (body frame: m/s, m/s, rad/s); CoT in J/(kg·m)")
    return rows, "\n".join(lines)
