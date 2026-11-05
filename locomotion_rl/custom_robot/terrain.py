"""Fractal terrain for the custom robot's rough-terrain task.

Spectral synthesis: white noise is shaped in the Fourier domain so that the height amplitude at spatial frequency f falls
off as f^-(hurst+1), within a band from `max_wavelength` (the longest hill; longer ones would just tilt the whole robot)
down to `min_wavelength` (the finest bumps; the feet are 8 mm wide, so a few cm). A small Hurst exponent leaves much of
the energy at short wavelengths (spiky), a large one is smooth. The result is scaled to the requested rms height.

Everything about the shape is a parameter of `fractal_heights`; `stats` reports what the feet will actually meet, in
units that matter for this robot (a stride is ~30 mm, the swing clearance of the learned gait ~3 mm, the foot radius 4 mm).
"""
import numpy as np

DEFAULT = dict(
    density=200.0,          # heightfield points per metre (spacing 5 mm; 2 triangles per cell = 2 * density^2 per m^2)
    half_extent=5.0,        # m, MuJoCo hfield radius
    rms=0.0025,             # m, rms height of the field
    hurst=0.5,
    min_wavelength=0.02,    # m
    max_wavelength=0.30,    # m
    seed=0,
)


def grid_points(density, half_extent):
    """Points per side of the heightfield. The first and last points sit on the edges, so the spacing is exactly
    2 * half_extent / (points - 1), which is 1 / density when the extent is a whole number of spacings."""
    return int(round(2 * half_extent * density)) + 1


def spacing(density, half_extent):
    """Distance between neighbouring heightfield points, m."""
    return 2 * half_extent / (grid_points(density, half_extent) - 1)


def fractal_heights(density=200.0, half_extent=5.0, rms=0.0025, hurst=0.5, min_wavelength=0.02, max_wavelength=0.30, seed=0):
    """Heights in metres, shape (n, n) with n = grid_points(), minimum 0. Row i is y = -half_extent + i * spacing, column j is x."""
    size = grid_points(density, half_extent)
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(size, d=spacing(density, half_extent))
    f = np.hypot(*np.meshgrid(fx, fx))
    f_lo, f_hi = 1.0 / max_wavelength, 1.0 / min_wavelength
    with np.errstate(divide="ignore"):
        amp = np.where(f > 0, f ** -(hurst + 1.0), 0.0)
    band = np.exp(-(f / f_hi) ** 4) * (1.0 - np.exp(-(f / f_lo) ** 4))        # smooth band-pass
    h = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * amp * band).real
    h = (h - h.mean()) * (rms / h.std())
    return h - h.min()


def _perlin(x, y, seed):
    """2D gradient (Perlin) noise at lattice coordinates x, y (arrays), roughly in [-1, 1]."""
    perm = np.random.default_rng(seed).permutation(256)
    perm = np.concatenate([perm, perm])
    grads = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    xf, yf = x - xi, y - yi
    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    u, v = fade(xf), fade(yf)

    def corner(ix, iy, dx, dy):
        g = grads[perm[(perm[ix & 255] + iy) & 255] & 7]
        return g[..., 0] * dx + g[..., 1] * dy

    n00, n10 = corner(xi, yi, xf, yf), corner(xi + 1, yi, xf - 1, yf)
    n01, n11 = corner(xi, yi + 1, xf, yf - 1), corner(xi + 1, yi + 1, xf - 1, yf - 1)
    return np.sqrt(2.0) * ((n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v)


def perlin_fbm_heights(density=200.0, half_extent=5.0, frequency=2.0, amplitude=0.01, octaves=2, lacunarity=2.0,
                       gain=0.25, seed=0):
    """The paper's terrain vocabulary (RaiSim's fractal generator): fractal Brownian motion of Perlin noise.

    frequency: base lattice cycles per metre (RaiSim's documentation); octave k runs at frequency * lacunarity^k with
    amplitude gain^k. amplitude: metres, multiplies the noise normalised to roughly [0, 1] (RaiSim's `zScale`). It is a
    nominal scale: the peak-to-peak comes out about 1.2x larger because the noise slightly exceeds [0, 1].
    This is a reimplementation of the described parameters, not RaiSim's code, so it matches in character not bit for bit.
    """
    n = grid_points(density, half_extent)
    axis = np.linspace(-half_extent, half_extent, n)
    x, y = np.meshgrid(axis, axis)
    total, weight = np.zeros((n, n)), 0.0
    for k in range(octaves):
        total += gain ** k * _perlin(x * frequency * lacunarity ** k + 17.3 * k, y * frequency * lacunarity ** k + 9.1 * k, seed + k)
        weight += gain ** k
    h = amplitude * (0.5 + 0.5 * total / weight)
    return h - h.min()


# The paper (Fu et al. 2021, RaiSim's fractal generator) uses 2 octaves, lacunarity 2.0, gain 0.25 for both of its terrains, which
# differ only in frequency (10 -> 20) and amplitude (0.23 -> 0.27 m): rough = twice the frequency, 1.17x the amplitude. Those
# three settings are unit-free and are kept. Taken literally, frequency/amplitude give slopes of 51 and 69 degrees on average
# (cycles per metre, metres), which cannot be what an A1 walked on, so the units there are not what they appear to be, and the
# two are re-chosen for this robot from the paper's figure instead: against the A1's foot (a ~4-5 cm cap, 37 px in the figure)
# the terrain facets are 1-2 foot widths across and the bumps about one foot width tall. Our foot is 8 mm wide, so: facets of
# 8-16 mm (hence density 100 points/m, which also trains ~1.8x faster than 200) and ~5-8 mm of relief per two foot lengths.
# That estimate comes from one perspective image, good to maybe +-50 %; treat the amplitudes as the first points of a sweep.
PRESETS = dict(
    structured=dict(kind="perlin", density=100.0, frequency=12.5, amplitude=0.0170, octaves=2, lacunarity=2.0, gain=0.25),
    unstructured=dict(kind="perlin", density=100.0, frequency=25.0, amplitude=0.0200, octaves=2, lacunarity=2.0, gain=0.25),
)


def block_heights(density=200.0, half_extent=5.0, block=0.10, rms=0.0031, seed=0):
    """Discrete terrain: a grid of square cuboids, `block` m across the top, each with its own random height.

    Heights are independent normal draws with standard deviation `rms`, shifted so the lowest block is 0. The default rms
    is that of the fractal rough scene (3.1 mm, 25 mm peak-to-peak over its 1M points, which is also what ~10^4 normal
    draws span), so the two scenes differ in structure, not in amplitude.

    The result is a heightfield sampled like the others (same shape, same hfield machinery in MJX / Warp). Heightfield
    samples are joined linearly, so a vertical cuboid wall is one cell wide: block k owns the samples whose position lies in
    [k * block, (k + 1) * block), and the step to the next block takes one `1 / density` cell (5 mm at 200 per m). That makes
    the flat top `block - 1 / density` wide plus the slanted wall, and the wall slope for a typical step of ~4 mm about
    40 deg at 200 per m, 22 deg at 100 per m. `block` must be a whole number of cells.
    """
    n = grid_points(density, half_extent)
    cells = block * density
    if abs(cells - round(cells)) > 1e-6:
        raise ValueError(f"block {block} m is not a whole number of cells at {density:g} points per m")
    nblocks = int(round(2 * half_extent / block))
    levels = np.random.default_rng(seed).normal(0.0, rms, (nblocks, nblocks))
    idx = np.minimum(np.arange(n) // int(round(cells)), nblocks - 1)          # block of each sample; the last sample joins the last block
    h = levels[np.ix_(idx, idx)]                                              # row i is y, column j is x, as in the other generators
    return h - h.min()


def heights(kind="spectral", **kw):
    """Dispatch on the generator kind; kw are that generator's parameters (density, half_extent, seed, ...)."""
    if kind == "spectral":
        return fractal_heights(**kw)
    if kind == "perlin":
        return perlin_fbm_heights(**kw)
    if kind == "blocks":
        return block_heights(**kw)
    raise ValueError(f"unknown terrain kind {kind!r}")


def to_png_levels(h, elevation):
    """8-bit levels that MuJoCo maps to 0..elevation metres."""
    return np.round(255.0 * h / elevation).astype(np.uint8)


def stats(h, spacing, window=0.03):
    """What the feet meet: rms, peak-to-peak, slopes, and relief inside a `window` (one stride) neighbourhood."""
    k = max(int(round(window / spacing)), 2)
    gx, gy = np.gradient(h, spacing)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    blocks = h[: h.shape[0] // k * k, : h.shape[1] // k * k].reshape(h.shape[0] // k, k, h.shape[1] // k, k)
    relief = blocks.max((1, 3)) - blocks.min((1, 3))
    return dict(rms=h.std(), p2p=np.ptp(h), slope_mean=slope.mean(), slope_p95=np.percentile(slope, 95), slope_max=slope.max(),
                relief_median=np.median(relief), relief_p95=np.percentile(relief, 95), window=k * spacing)
