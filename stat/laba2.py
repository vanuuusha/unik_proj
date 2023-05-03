import math

import numpy as np
from scipy.stats import norm, chi2, t

n = 50


def m_with_d(sample, sigma):
    q = norm.ppf(0.95)
    mean = sample.mean()
    start = mean - q * sigma/math.sqrt(n)
    stop = mean + q * sigma/math.sqrt(n)
    print(f'math with sigma {round(start, 5)} - {round(stop, 5)}')
    return start, stop

def m_without_d(sample):
    q = t.ppf(0.95, n-1)
    mean = sample.mean()
    desp = np.var(sample, ddof=1)
    sigma = math.sqrt(desp)
    start = mean - q * sigma/math.sqrt(n)
    stop = mean + q * sigma/math.sqrt(n)
    print(f'math without sigma {round(start, 5)} - {round(stop, 5)}')
    return start, stop

def d_with_m(sample, math_wait):
    q1 = chi2.ppf(0.95, n)
    q2 = chi2.ppf(0.05, n)
    desp = np.var(sample, ddof=1)
    start = (n-1) * desp / q1
    stop = (n-1) * desp / q2
    print(f'sigma^2 with math {round(start, 5)} - {round(stop, 5)}')
    return start, stop

def d_without_m(sample):
    q1 = chi2.ppf(0.95, n-1)
    q2 = chi2.ppf(0.05, n-1)
    desp = np.var(sample, ddof=1)
    start = (n-1) * desp / q1
    stop = (n-1) * desp / q2
    print(f'sigma^2 withot math {round(start, 5)} - {round(stop, 5)}')
    return start, stop


if __name__ == '__main__':
    sigma = 3
    math_wait = 2
    sample = np.random.normal(math_wait, sigma, n)
    m_w_d = m_with_d(sample, sigma)
    d_w_m = d_with_m(sample, math_wait)
    d_no_m = d_without_m(sample)
    m_no_d = m_without_d(sample)


    counts = [0, 0, 0, 0]
    for i in range(200):
        sample = np.random.normal(math_wait, sigma, n)
        now_math = sample.mean()
        now_desp = np.var(sample, ddof=1)
        if m_w_d[0] < now_math < m_w_d[1]:
            counts[0] += 1
        if d_w_m[0] < now_desp < d_w_m[1]:
            counts[1] += 1
        if d_no_m[0] < now_desp < d_no_m[1]:
            counts[2] += 1
        if m_no_d[0] < now_math < m_no_d[1]:
            counts[3] += 1

    count = [i /200 for i in counts]
    print(count)