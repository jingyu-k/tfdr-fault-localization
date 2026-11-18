import numpy as np
from scipy.signal import correlate, correlation_lags

FS = 5e9          # 샘플링 주파수 (5 GHz)
V = 2e8           # 케이블 속 신호 속도 (m/s)
T0 = 50e-9        # 처프를 보내는 시각
DURATION = 1e-6   # 기록 길이 (1 마이크로초)


def time_axis(fs=FS, duration=DURATION):
    return np.arange(0, duration, 1/fs)


def gaussian_chirp(t, t_center, fc=200e6, bandwidth=200e6, sigma=15e-9):
    """가우시안 포락선을 씌운 처프"""
    tau = t - t_center
    k = bandwidth / (4 * sigma)
    envelope = np.exp(-tau**2 / (2 * sigma**2))
    return envelope * np.cos(2*np.pi*(fc*tau + 0.5*k*tau**2))


def simulate_received(t, faults, noise_std=0.2, rng=None, **chirp_kwargs):
    """faults: [(고장 거리 m, 반사 계수), ...] 목록
    chirp_kwargs: fc, bandwidth, sigma 등 처프 설정을 바꿀 때 사용"""
    if rng is None:
        rng = np.random.default_rng()
    ref = gaussian_chirp(t, T0, **chirp_kwargs)
    rx = ref.copy()
    for d, gamma in faults:
        rx += gamma * gaussian_chirp(t, T0 + 2*d/V, **chirp_kwargs)
    rx += noise_std * rng.standard_normal(len(t))
    return ref, rx


def cross_correlate(rx, ref, fs=FS):
    c = correlate(rx, ref, mode='full')
    lags = correlation_lags(len(rx), len(ref), mode='full') / fs
    return c, lags


def estimate_distance(c, lags, min_delay=100e-9):
    """보낸 신호 자체(0 ns 근처)를 제외하고 가장 큰 봉우리로 거리 추정"""
    mask = lags > min_delay
    idx = np.argmax(np.abs(c[mask]))
    return V * lags[mask][idx] / 2