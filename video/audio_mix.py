"""Procedural soundtrack + SFX + voice ducking for the explainer video.

All music is synthesized here (no samples -> no licensing issues).
Arrangement builds with the story:
  scene 0 pad only (mysterious) -> 1 + bass -> 2 + arpeggio -> 3-4 + drums (energy)
  -> 5 break, then a big hit on the title reveal and a sustained final chord.
"""
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
BPM = 112
BEAT = 60 / BPM
BAR = 4 * BEAT
# i - VI - III - VII in A minor: Am F C G
CHORDS = [(57, 60, 64), (53, 57, 60), (55, 60, 64), (55, 59, 62)]
ROOTS = [45, 41, 48, 43]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def soft_saw(f, t, nh=6):
    out = np.zeros_like(t)
    for k in range(1, nh + 1):
        out += np.sin(2 * np.pi * k * f * t) * (0.62 ** k) / k
    return out


def env_ar(n, a, r):
    e = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    if na:
        e[:na] = np.linspace(0, 1, na)
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def add(buf, start_s, sig):
    i = int(start_s * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]


def bandnoise(n, lo, hi, rng):
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, rng.standard_normal(n))


def smooth(x, k):
    """O(N) centred moving average (np.convolve with a long box kernel is O(N*k))."""
    k = max(1, int(k))
    c = np.cumsum(np.concatenate([np.zeros(k // 2 + 1), x, np.zeros(k)]))
    return (c[k:k + len(x)] - c[:len(x)]) / k


def section_gain(N, scenes, active, ramp=0.6):
    """Per-sample gain, 1 inside scenes listed in `active` (idx or (idx, frac0, frac1))."""
    g = np.zeros(N)
    for spec in active:
        idx, f0, f1 = (spec, 0.0, 1.0) if isinstance(spec, int) else spec
        sc = scenes[idx]
        a = int((sc["start"] + f0 * sc["T"]) * SR)
        b = int((sc["start"] + f1 * sc["T"]) * SR)
        g[a:b] = 1.0
    return smooth(g, ramp * SR)


def build_music(total, scenes, rng, split):
    N = int(total * SR) + SR
    t_full = np.arange(N) / SR
    pad = np.zeros(N)
    bass = np.zeros(N)
    arp = np.zeros(N)
    kick = np.zeros(N)
    hat = np.zeros(N)

    nbars = int(total / BAR) + 2
    for b in range(nbars):
        t0 = b * BAR
        ci = b % 4
        # pad
        n = int((BAR + 0.8) * SR)
        tt = np.arange(n) / SR
        seg = np.zeros(n)
        for m in CHORDS[ci]:
            for det in (-0.06, 0.0, 0.06):
                seg += soft_saw(hz(m + det), tt + rng.uniform(0, 1))
        add(pad, t0, seg * env_ar(n, 0.5, 0.9) / 9)
        # bass: eighth-note pulse
        for e in range(8):
            n = int(BEAT / 2 * SR)
            tt = np.arange(n) / SR
            f = hz(ROOTS[ci] - 12 if e % 2 == 0 else ROOTS[ci])
            s = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 7)
            add(bass, t0 + e * BEAT / 2, s * env_ar(n, 0.004, 0.02))
        # arp: 16ths over chord tones
        tones = [m + o for o in (12, 24) for m in CHORDS[ci]]
        pattern = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 2]
        for s16, p in enumerate(pattern):
            n = int(0.35 * SR)
            tt = np.arange(n) / SR
            f = hz(tones[p])
            s = (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 11)
            add(arp, t0 + s16 * BEAT / 4, s * env_ar(n, 0.003, 0.05))
        # drums
        for bt in range(4):
            n = int(0.35 * SR)
            tt = np.arange(n) / SR
            ph = 2 * np.pi * np.cumsum(45 + 95 * np.exp(-tt * 28)) / SR
            add(kick, t0 + bt * BEAT, np.sin(ph) * np.exp(-tt * 9))
            n2 = int(0.06 * SR)
            hn = bandnoise(n2 + 200, 6500, 15000, rng)[200:] * np.exp(-np.arange(n2) / SR * 60)
            add(hat, t0 + bt * BEAT + BEAT / 2, hn)

    last = len(scenes) - 1
    g_pad = section_gain(N, scenes, list(range(last)) + [(last, 0, split)], ramp=1.5)
    g_bass = section_gain(N, scenes, [1, 2, 3, 4])
    g_arp = section_gain(N, scenes, [(2, 0.25, 1), 3, 4, (last, 0, split)])
    g_drum = section_gain(N, scenes, [(2, 0.85, 1), 3, 4], ramp=0.3)

    # intro fade-in
    fade = np.clip(t_full / 2.5, 0, 1)
    music = fade * (0.9 * pad * g_pad + 0.55 * bass * g_bass + 0.22 * arp * g_arp)
    drums = 0.75 * kick * g_drum + 0.10 * hat * g_drum

    # final chord swell on the title reveal (C major add9 -> warm resolve)
    sc = scenes[last]
    t_hit = sc["start"] + split * sc["T"]
    n = int((total - t_hit + 0.5) * SR)
    tt = np.arange(n) / SR
    chord = np.zeros(n)
    for m in (48, 55, 60, 64, 67, 74):
        for det in (-0.07, 0.07):
            chord += soft_saw(hz(m + det), tt, nh=5)
    chord *= np.minimum(1, tt / 0.08) * np.exp(-tt * 0.35) / 12
    add(music, t_hit, 1.1 * chord)
    return music, drums, t_hit


def build_sfx(total, scenes, t_alert_abs, t_hit, rng):
    N = int(total * SR) + SR
    sfx = np.zeros(N)
    # whoosh into every scene change
    for sc in scenes[1:]:
        n = int(0.9 * SR)
        tt = np.arange(n) / SR
        w = bandnoise(n, 300, 5000, rng)
        e = np.sin(np.pi * tt / tt[-1]) ** 2
        add(sfx, sc["start"] - 0.55, 0.22 * w * e)
    # riser into the capabilities scene
    sc = scenes[3]
    n = int(2.2 * SR)
    tt = np.arange(n) / SR
    r = bandnoise(n, 800, 9000, rng) * (tt / tt[-1]) ** 2
    tone = np.sin(2 * np.pi * np.cumsum(200 + 600 * (tt / tt[-1]) ** 2) / SR) * (tt / tt[-1]) ** 2
    add(sfx, sc["start"] - 2.2, 0.16 * r + 0.06 * tone)
    # alert: two triple-beeps
    for k in range(2):
        for j in range(3):
            n = int(0.11 * SR)
            tt = np.arange(n) / SR
            s = np.sign(np.sin(2 * np.pi * 988 * tt)) * 0.5 + np.sin(2 * np.pi * 1976 * tt) * 0.3
            add(sfx, t_alert_abs + k * 0.75 + j * 0.16, 0.10 * s * env_ar(n, 0.005, 0.03))
    # title impact: sub boom + noise crash
    n = int(2.5 * SR)
    tt = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(30 + 60 * np.exp(-tt * 6)) / SR) * np.exp(-tt * 2.2)
    crash = bandnoise(n, 2000, 14000, rng) * np.exp(-tt * 2.8)
    add(sfx, t_hit, 0.6 * boom + 0.10 * crash)
    return sfx


def reverb(x, rng, secs=2.2):
    n = int(secs * SR)
    tt = np.arange(n) / SR
    irs = [rng.standard_normal(n) * np.exp(-tt * 3.2) for _ in range(2)]
    irs = [ir / np.sqrt((ir ** 2).sum()) for ir in irs]
    return [fftconvolve(x, ir)[: len(x)] for ir in irs]


def voice_envelope(voice):
    a = np.abs(voice)
    env = smooth(a, 0.25 * SR)
    env /= env.max() + 1e-9
    return np.clip(env * 4, 0, 1)


def mix(voice, scenes, total, t_alert_abs, title_frac):
    """voice: mono float array in [-1, 1] at SR. Returns stereo int16 (N, 2)."""
    rng = np.random.default_rng(11)
    music, drums, t_hit = build_music(total, scenes, rng, title_frac)
    sfx = build_sfx(total, scenes, t_alert_abs, t_hit, rng)
    N = len(voice)
    music, drums, sfx = music[:N], drums[:N], sfx[:N]

    wl, wr = reverb(music, rng)
    music_l = 0.8 * music + 0.35 * wl
    music_r = 0.8 * music + 0.35 * wr

    duck = 1 - 0.6 * voice_envelope(voice)
    bed_l = (music_l + drums) * duck
    bed_r = (music_r + drums) * duck
    # level: music bed peaks ~ -12 dB relative to voice
    bed_peak = max(np.abs(bed_l).max(), np.abs(bed_r).max()) + 1e-9
    bed_l *= 0.30 / bed_peak
    bed_r *= 0.30 / bed_peak
    sfx_pk = np.abs(sfx).max() + 1e-9
    sfx *= 0.45 / sfx_pk
    v = voice / (np.abs(voice).max() + 1e-9) * 0.85

    L = v + bed_l + sfx
    R = v + bed_r + sfx
    st = np.stack([L, R], 1)
    st /= max(1.0, np.abs(st).max() / 0.97)
    # fade out tail
    nf = int(1.5 * SR)
    st[-nf:] *= np.linspace(1, 0, nf)[:, None]
    return (st * 32767).astype(np.int16)
