import math
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats


class State:
    def __init__(self, my_lambda, name):
        self.name = name
        self.my_lambda = my_lambda
        self.another_states = []

    def stay(self):
        return np.random.exponential(1/self.my_lambda)

    def add_another_state(self, state, ver):
        self.another_states.append((state, ver))

    def choice_new_state(self):
        num = np.random.random()
        if num <= self.another_states[0][1]:
            return self.another_states[0][0]
        return self.another_states[1][0]

    def __repr__(self):
        return self.name


class MarkovProcces:
    def __init__(self, ver_mat, lamdas, t, do_draw=False):
        self.do_draw = do_draw
        self.ver_mat = ver_mat
        self.max_time = t
        self.now_time = 0
        self.all_time = []
        self.all_states = []
        self.states = [State(lamdas[0], '0'), State(lamdas[1], '1'), State(lamdas[2], '2')]
        self.now_state = np.random.choice(self.states)
        self.add_vers_to_states()
        self.start_process()

    def add_vers_to_states(self):
        for i in range(3):
            can_be = list(range(3))
            can_be.pop(i)
            for j in can_be:
                self.states[i].add_another_state(self.states[j], self.ver_mat[i][j])

    def draw(self):
        y = list(map(int, [i.name for i in self.all_states]))
        x = self.all_time

        for i in range(1, len(x)):
            plt.hlines(y[i - 1], x[i-1], x[i])
            plt.vlines(x[i], y[i - 1], y[i])

        plt.hlines(y[-1], x[-1], self.max_time)
        plt.show()


    def start_process(self):
        while True:
            self.all_time.append(self.now_time)
            self.all_states.append(self.now_state)
            self.now_time += self.now_state.stay()
            if self.now_time > self.max_time:
                break
            self.now_state = self.now_state.choice_new_state()
        if self.do_draw:
            self.draw()

    def calculate_all_time(self):
        states = list(map(int, [i.name for i in self.all_states]))
        res = {i: 0 for i in range(3)}
        for i in range(len(self.all_time) - 1):
            res[states[i]] += self.all_time[i+1] - self.all_time[i]
        res[states[-1]] += self.max_time - self.all_time[-1]
        return res


def make_ver_matrix(in_mat):
    ver_mat = []
    for i in range(3):
        temp = []
        for j in range(3):
            if i == j:
                temp.append(0)
            else:
                temp.append(in_mat[i, j] / -in_mat[i, i])
        ver_mat.append(temp)
    return ver_mat


def make_markov(t, in_mat, draw):
    ver_mat = make_ver_matrix(in_mat)
    MarkovProcces(ver_mat, [-in_mat[0, 0], -in_mat[1, 1], -in_mat[2, 2]], t, do_draw=draw)


def make_static_ver(transition_matrix, t=1000):
    Q = transition_matrix
    # решаем уравнение Q * pi = 0 для pi
    w, v = np.linalg.eig(Q.T)
    pi = v[:, np.isclose(w, 0)]
    # нормируем вектор pi
    pi = pi / pi.sum()
    print('Теоретические стационарные вероятности:', pi)

    ver_mat = make_ver_matrix(transition_matrix)
    proc = MarkovProcces(ver_mat, [-transition_matrix[0, 0], -transition_matrix[1, 1], -transition_matrix[2, 2]], t=t, do_draw=False)
    emper = proc.calculate_all_time()
    for i in emper.keys():
        emper[i] /= t
    print('Эмпирические стационарные вероятности: ', emper.values())


def check_cpt(transition_matrix, t=100):
    ver_mat = make_ver_matrix(in_mat)
    all_emper_vers = {0: [], 1: [], 2: []}
    for i in range(1000):
        proc = MarkovProcces(ver_mat, [-in_mat[0, 0], -in_mat[1, 1], -in_mat[2, 2]], t, do_draw=False)
        emper = proc.calculate_all_time()
        for k in emper.keys():
            emper[k] /= t
        for k in all_emper_vers:
            all_emper_vers[k].append(emper[k])
    draw_hist(all_emper_vers[0])
    draw_hist(all_emper_vers[1])
    draw_hist(all_emper_vers[2])



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

    mu, std = stats.norm.fit(sample)
    print(mu, std)
    x = np.linspace(-2, 2, num=1000)  # создаем массив значений x
    y = stats.norm.pdf(x, mu, std) * width  # вычисляем плотность вероятности стандартного нормального распределения
    plt.plot(x, y, c='r')  # строим график

    ax.set_title('Гистограмма')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    in_mat = np.array([[-0.3, 0.1, 0.2],
                        [0.15, -0.2, 0.05],
                        [0.15, 0.25, -0.4]])
    #make_markov(t=100, in_mat=in_mat, draw=True)
    # make_static_ver(in_mat, t=10000)
    # check_cpt(in_mat)