import math

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def find_sn(t, X, is_last, return_sn=False):
    S_n = 0
    n = 0
    while S_n <= t:
        S_n += X[n]
        n += 1
    n -= 1
    S_n -= X[n]
    if is_last:
        print(S_n / n)
    if return_sn:
        return S_n, n
    return n


def gen_proc(mu, std, t, ax):
    n = []
    X = np.random.normal(mu, std, size=100000)
    for t_now in t:
        n.append(find_sn(t_now, X, is_last=(t_now == t[-1])))
    draw(n, t, ax)


def draw(n, t, ax):
    col = (np.random.random(), np.random.random(), np.random.random())
    ax.plot(t, n, c=col)


def gen_all_tract(n, mu, std):
    fig = plt.figure()
    ax = fig.add_subplot()

    for _ in range(n):
        gen_proc(mu, std, np.arange(0, 100, 0.1), ax)

    ax.set_xlabel('Время')
    ax.set_ylabel('n')
    ax.grid()
    plt.show()


def give_S_n_and_n(count, mu, std):
    S_n = []
    n = []
    for _ in range(count):
        X = np.random.normal(mu, std, size=100000)
        S_n_now, n_now = find_sn(99.9, X, is_last=False, return_sn=True)
        S_n.append(S_n_now)
        n.append(n_now)
    return S_n, n


def draw_hist(sample):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_bins = 10
    intervals = np.linspace(min(sample), max(sample), n_bins) # начало каждого интервала
    frequency = np.array([len([value for value in sample if intervals[i] <= value < intervals[i + 1]]) for i in range(n_bins-1)])
    frequency = frequency / len(sample) # частоты
    labels = [(intervals[i] + intervals[i + 1]) / 2 for i in range(n_bins-1)] # середины столбцов
    width = labels[1] - labels[0] # ширина одного столбца

    ax.bar(labels, frequency, width)

    x = np.linspace(-5, 5, num=1000)  # создаем массив значений x
    y = stats.norm.pdf(x) * width  # вычисляем плотность вероятности стандартного нормального распределения
    plt.plot(x, y, c='r')  # строим график

    ax.set_title('Гистограмма')
    plt.grid()
    plt.show()


def calculate_chi_2(sample):
    n_bins = 20
    n =len(sample)
    bin_edges = np.linspace(np.min(sample), np.max(sample), n_bins)
    p_i = np.array([stats.norm.cdf(bin_edges[i]) - stats.norm.cdf(bin_edges[i-1]) for i in range(1, n_bins)])
    n_p_i = n*p_i

    h_n_i = np.array(
        [len([value for value in sample if bin_edges[i- 1] <= value < bin_edges[i]]) for i in range(1, n_bins)])

    chi_2 = sum((h_n_i - n_p_i) ** 2 / n_p_i)

    df = n_bins - 1  # степени свободы
    alpha = 0.95  # уровень значимости
    c = stats.chi2.ppf(alpha, df)
    print(chi_2, c)


if __name__ == '__main__':
    mu = 3
    std = 1
    # gen_all_tract(n=10, mu=mu, std=std)

    S_n, n = give_S_n_and_n(count=500, mu=mu, std=std)
    S_n = [(S_n[i] - mu * n[i]) / (std * math.sqrt(n[i])) for i in range(len(S_n))]
    draw_hist(S_n)
    calculate_chi_2(S_n)