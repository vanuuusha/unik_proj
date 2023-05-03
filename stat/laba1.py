import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data = '1968.492 41.1261 0.0857 60.7957 2216.473 41.1261 0.0857 41.6195 2216.722 809.6731 0.1788 43.5278 2221.238 809.6731 0.1788 34.2727 2221.242 310.6746 0.1813 34.2839 2230.852 310.6746                0.1813 49.5744 2230.862 201.2641 0.1654 49.6587 2240.280 201.2641 0.1654 48.5889 2240.289 395.0617 0.1730 48.6801 2249.526 395.0617 0.1730 63.8574 2249.535 138.6235 0.1413 63.9331 2251.817 138.6235 0.1413 73.2354 2251.819 139.1573 0.1586 73.2369 2260.172 139.1573 0.1586 73.9546 2260.180 119.5031 0.1353 73.9877 2269.059 119.5031 0.1353 78.7947 2269.068 292.8690 0.1453 78.8402 2277.819 292.8690 0.1453 88.6888 2277.828 220.0500 0.1566 88.6946 2286.453            220.0500 0.1566 70.0894 2286.462 92.8746 0.1440 70.1490 2294.964 92.8746 0.1440 80.3080 2294.972 70.8309 0.1499 80.2755 2303.363 70.8309 0.1499 71.7327 2303.372 133.8180 0.1567 71.6800 2311.662 133.8180 0.1567 73.2779 2311.670 127.8850 0.1502 73.2854 2319.864 127.8850 0.1502 84.1279 2319.872 28.8911 0.1270 84.2226 2321.135 28.8911 0.1270 99.8326 2321.136 22.8408 0.1323 99.8410 2329.262 22.8408 0.1323 92.3621 2329.270 55.3365 0.1447 92.3515 2337.311 55.3365 0.1447 90.0307 2337.319 50.1856 0.1411 90.0423 2345.281 50.1856 0.1411 91.4409 2345.289 19.7099 0.1337 91.4625 2353.179 19.7099 0.1337 90.7896 2353.187 23.8111 0.1328 90.7789 2361.009 23.8111 0.1328 98.0200 2361.016 24.9681 0.1285 97.9753 2361.660 24.9681 0.1285 94.2984 2361.660 58.0710 0.1401 94.2948 2366.887 58.0710 0.1401 80.8433 2366.893 80.6367 0.1548 80.8558 2374.759 80.6367 0.1548 63.0324 2374.767 159.4482 0.1680 63.0001 2382.580 159.4482 0.1680 81.9761 2382.587 150.2006 0.1592 82.0260 2390.350 150.2006 0.1592 49.0234 2390.358 147.3873 0.1607 49.0414 2398.074 147.3873 0.1607 66.3515 2398.082 198.8038 0.1425 66.3417 2405.749 198.8038 0.1425 79.2673 2405.757 261.4384 0.1503 79.2162 2413.385 261.4384 0.1503 44.0515 2413.392 383.6540 0.1799 43.9904 2420.979 383.6540 0.1799 69.4734 2420.987 89.1650 0.1239 69.5303 2421.380 89.1650 0.1239 73.3943 2421.380 45.5443 0.1278 73.3982 2425.361 45.5443 0.1278 120.4646 2425.365 9.5750 0.1079 120.4943 2433.004 9.5750 0.1079 115.9249 2433.011 22.2311 0.1111 115.9065 2440.719 22.2311 0.1111 112.1906 2440.727 11.6592 0.1126 112.1831 2448.403 11.6592 0.1126 113.2050 2448.411 13.0283 0.1096 113.1893 2456.067 13.0283 0.1096 113.2159 2456.075 26.1558 0.1200 113.1746 2463.703 26.1558 0.1200 106.1164'.split()
depth = np.array([data[i] for i in range(len(data)) if i % 4 == 0])
penetration = np.array([data[i] for i in range(len(data)) if i % 4 == 1])
porosity = np.array([data[i] for i in range(len(data)) if i % 4 == 2])
gamma_ray = np.array([data[i] for i in range(len(data)) if i % 4 == 3])


def calculate_sample(sample):
    res = {}
    res['mean'] = sample.mean()
    res['dispersion'] = np.var(sample, ddof=1)
    res['standart'] = np.sqrt(res['dispersion'])
    n = sample.shape[0]
    temp = math.floor(n / 4)
    res['25'] = sample[temp]
    if n % 2 == 1:
        temp = (n - 1) / 2 - 1
        res['50'] = sample[temp]
    else:
        temp = int(n / 2) - 1
        temp_2 = int(n / 2)
        res['50'] = (sample[temp] + sample[temp_2]) / 2
    temp = math.floor(3 * n / 4)
    res['75'] = sample[temp]
    res['min'] = res['25'] - 1.5 * (res['75'] - res['25'])
    res['max'] = res['75'] + 1.5 * (res['75'] - res['25'])
    return res


def calculate_cov(sample_1, res_1, sample_2, res_2):
    res = {}
    res['cov'] = 1 / (sample_1.shape[0] - 1) * np.sum((sample_1 - res_1['mean'])*(sample_2 - res_2['mean']))
    res['r'] = res['cov'] / (res_1['standart'] * res_2['standart'])
    return res


def first(depth, penetration, sample_size=20):
    np.random.seed(42)
    sample_indexes = np.random.randint(0, len(penetration), sample_size)
    sample_depth = np.sort(np.array(depth, dtype=float)[sample_indexes])
    sample_penetration = np.sort(np.array(penetration, dtype=float)[sample_indexes])

    depth_res = calculate_sample(sample_depth)
    penetration_res = calculate_sample(sample_penetration)

    cov_res = calculate_cov(sample_depth, depth_res, sample_penetration, penetration_res)

    #draw_gistogram_2_sample(sample_depth, sample_penetration, 'Гистограмма пористость-глубина')
    #draw_gistogram_1_sample(sample_depth, name='глубина')
    #draw_gistogram_1_sample(sample_penetration, name='проницаемость')
    #draw_boxplot(depth_res, 'Глубина')
    #draw_boxplot(penetration_res, 'Проницаемость')
    sample_raspr(sample_depth, 'глубина')
    sample_raspr(sample_penetration, 'проницаемость')

    print(depth_res)
    print(penetration_res)
    print(cov_res)


def second(sample_size):
    np.random.seed(42)
    sample_1 = np.random.normal(2, 4, sample_size)
    np.random.seed(40)
    sample_2 = np.random.normal(2, 4, sample_size)


    sample_1_res = calculate_sample(np.sort(sample_1))
    sample_2_res = calculate_sample(np.sort(sample_2))

    cov_res = calculate_cov(sample_1, sample_1_res, sample_2, sample_2_res)

    sample_1 = np.sort(sample_1)
    sample_2 = np.sort(sample_2)

    # draw_gistogram_2_sample(sample_1, sample_2, f'Гистограмма 1-2 n={sample_size}', axes_names=('x', 'y', 'z'))
    # draw_gistogram_1_sample(sample_1, name=f'Выборка 1 n={sample_size}')
    # draw_gistogram_1_sample(sample_2, name=f'Выборка 2 n={sample_size}')
    # draw_boxplot(sample_1_res, f'Выборка 1 n={sample_size}')
    # draw_boxplot(sample_2_res, f'Выборка 2 n={sample_size}')
    # sample_raspr(sample_1, f'Выборка 1 n={sample_size}')
    # sample_raspr(sample_2, f'Выборка 2 n={sample_size}')

    print(sample_1_res)
    print(sample_2_res)
    print(cov_res)

def third(sample_size):
    np.random.seed(42)
    sample_1 = np.random.gamma(2, 4, sample_size)
    np.random.seed(40)
    sample_2 = np.random.gamma(2, 4, sample_size)


    sample_1_res = calculate_sample(np.sort(sample_1))
    sample_2_res = calculate_sample(np.sort(sample_2))

    cov_res = calculate_cov(sample_1, sample_1_res, sample_2, sample_2_res)

    sample_1 = np.sort(sample_1)
    sample_2 = np.sort(sample_2)

    # draw_gistogram_2_sample(sample_1, sample_2, f'Гистограмма 1-2 n={sample_size}', axes_names=('x', 'y', 'z'))
    # draw_gistogram_1_sample(sample_1, name=f'Выборка 1 n={sample_size}')
    # draw_gistogram_1_sample(sample_2, name=f'Выборка 2 n={sample_size}')
    # draw_boxplot(sample_1_res, f'Выборка 1 n={sample_size}')
    # draw_boxplot(sample_2_res, f'Выборка 2 n={sample_size}')
    # sample_raspr(sample_1, f'Выборка 1 n={sample_size}')
    # sample_raspr(sample_2, f'Выборка 2 n={sample_size}')

    print(sample_1_res)
    print(sample_2_res)
    print(cov_res)


def sample_raspr(sample, name):
    frequency = 0
    y = []
    for i in range(len(sample)):
        frequency += 1 / len(sample)
        y.append(frequency)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('Эмпирическая функция распределения ' + name)
    plt.plot(sample, y)
    plt.grid()
    plt.show()


def draw_boxplot(res, name):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(name)
    ax.boxplot([res['min'], res['max'], res['25'], res['50'], res['75']])
    n = 0.5
    ax.annotate(f'Нижний квартиль ({round(res["25"], 2)})', xy=(1, res['25']), arrowprops=dict(facecolor='black', arrowstyle='->'),
                xytext=(n, res['25']))
    ax.annotate(f'Верхний квартиль ({round(res["75"], 2)})', xy=(1, res['75']), arrowprops=dict(facecolor='black', arrowstyle='->'),
                xytext=(n, res['75']))
    ax.annotate(f'Медиана ({round(res["50"], 2)})', xy=(1, res['50']), arrowprops=dict(facecolor='black', arrowstyle='->'),
                xytext=(n, res['50']))
    ax.annotate(f'Максимум выборки ({round(res["max"], 2)})', xy=(1, res['max']), arrowprops=dict(facecolor='black', arrowstyle='->'),
                xytext=(n, res['max']))
    ax.annotate(f'Минимум выборки ({round(res["min"], 2)})', xy=(1, res['min']), arrowprops=dict(facecolor='black', arrowstyle='->'),
                xytext=(n, res['min']))

    plt.show()


def draw_gistogram_2_sample(sample_1, sample_2, name, axes_names=('Глубина', 'Проницаемость', 'Частота')):
    fig = plt.figure()
    ax_3d = fig.add_subplot(projection='3d')
    plt.title(name)
    ax_3d.set_xlabel(axes_names[0])
    ax_3d.set_ylabel(axes_names[1])
    ax_3d.set_zlabel(axes_names[2])
    ax_3d.grid()

    intervals_1 = np.linspace(min(sample_1), max(sample_1), 10)
    frequency_1 = np.array([len([value for value in sample_1 if intervals_1[i] <= value < intervals_1[i+1] ]) for i in range(9)])
    labels_1 = [(intervals_1[i] + intervals_1[i+1]) / 2 for i in range(9)]
    width_1 = labels_1[1] - labels_1[0]

    intervals_2 = np.linspace(min(sample_2), max(sample_2), 10)
    frequency_2 = np.array([len([value for value in sample_2 if intervals_2[i] <= value < intervals_2[i + 1]]) for i in range(9)])
    labels_2 = [(intervals_2[i] + intervals_2[i + 1]) / 2 for i in range(9)]
    width_2 = labels_2[1] - labels_2[0]

    xpos, ypos = np.meshgrid(labels_1, labels_2)
    zpos = np.zeros_like(xpos)

    z_frequency = []
    for i in frequency_1:
        temp = []
        for j in frequency_2:
            temp.append(i + j)
        z_frequency.append(temp)

    z_frequency = np.array(z_frequency)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = zpos.flatten()
    z_frequency = z_frequency.flatten()

    ax_3d.bar3d(xpos, ypos, zpos, width_1, width_2, z_frequency)

    plt.show()


def draw_gistogram_1_sample(sample, name):
    fig = plt.figure() # ploting figure
    ax = fig.add_subplot(111)
    ax.hist(sample)
    plt.title('Гистограмма ' + name)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    #first(depth, penetration)
    second(1000)
    #third(1000)