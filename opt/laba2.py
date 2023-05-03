import numpy as np
import matplotlib.pyplot as plt


class SymplexTable:
    def __init__(self, A, b, c):
        first_line = [' '] + [f'-x{i}' for i in range(1, A.shape[1] + 1)] + ['b']
        last_line = ['L'] + [-i for i in c[0]] + [0]
        self.table = [first_line]
        for i in range(A.shape[0]):
            new_line = [f'y{i + 1}'] + [A[i, k] for k in range(A.shape[1])] + [b[i, 0]]
            self.table.append(new_line)

        self.table.append(last_line)
        self.rows = len(self.table)
        self.columns = len(self.table[0])
        self.print()

    def change_basic(self, row: int, column: int):
        permissive_elem = self.table[row][column]

        self.table[0][column], self.table[row][0] = '-' + self.table[row][0], self.table[0][column].replace('-', '')

        self.table[row][column] = 1
        for i in range(1, self.rows):
            if i == row:
                continue
            self.table[i][column] = -self.table[i][column]

        for i in range(1, self.rows):
            if i == row:
                continue
            for j in range(1, self.columns):
                if j == column:
                    continue
                self.table[i][j] = self.table[i][j] * permissive_elem + self.table[i][column] * self.table[row][
                    j]  # другой знак так как в актуальной

        for i in range(1, self.rows):
            for j in range(1, self.columns):
                self.table[i][j] = self.table[i][j] / permissive_elem

    def make_self_basic(self):
        for what_change_i in range(1, self.rows):  # проходим по всему первому столбцу и ищем там y
            what_change = self.table[what_change_i][0]
            if 'y' in what_change:
                for on_what_i in range(1, self.columns):  # проходим по всей первой строке и ищем там x
                    on_what = self.table[0][on_what_i]
                    if 'x' in on_what:
                        break
                self.change_basic(row=what_change_i, column=on_what_i)  # меняем x и y
                self.delete_column(on_what_i)  # удаляем колонку с мнимым базисом
                self.print()

    def delete_column(self, column_id):
        for i in range(self.rows):
            self.table[i].pop(column_id)
        self.columns -= 1

    def print(self):
        matrix = np.array(self.table)
        col_widths = [max([len(row[i]) if '.' not in row[i] else len(row[i].split('.')) + 6 for row in matrix]) for i in
                      range(len(matrix[0]))]  # max в кажом столбце

        res = ''
        for row in matrix:
            res += " | ".join(
                row[i].ljust(col_widths[i]) if '.' not in row[i] else str(round(float(row[i]), 3)).ljust(col_widths[i])
                for i in range(len(row))) + '\n'  # выравниваем до макс
        print(res)

    def draw_on_nonbasic(self):
        if self.columns != 4:
            print('Невозможно провести геометрическое решение в таком базисе')
            return
        name_x_1 = self.table[0][self.columns - 3][2:]
        name_x_2 = self.table[0][self.columns - 2][2:]

        plt.rcParams.update({'font.size': 22})
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(1, self.rows - 1):
            x1 = np.arange(-20, 20, 0.01)
            b = self.table[i][self.columns - 1]
            a1 = -self.table[i][self.columns - 3]
            a2 = -self.table[i][self.columns - 2]
            x2 = - (b + a1 * x1) / a2
            ax.plot(x1, x2, '-',
                    label=f'{round(a1, 3)}*$x_{name_x_1}$ + ({round(a2, 3)})*$x_{name_x_2}$ + ({round(b, 3)}) $\geq$ 0')

            if a2 > 0:
                ax.fill_between(x1, x2, np.zeros_like(x2) + 100, alpha=0.5)
            else:
                ax.fill_between(x1, x2, np.zeros_like(x2) - 100, alpha=0.5)
        x1 = np.arange(-2, 2, 0.01)
        b = self.table[-1][self.columns - 1]
        a1 = -self.table[-1][self.columns - 3]
        a2 = -self.table[-1][self.columns - 2]
        x2 = 125 - (b + a1 * x1) / a2
        ax.plot(x1, x2, '-',
                label=f'{round(a1, 3)}*$x_{name_x_1}$ + ({round(a2, 3)})*$x_{name_x_2}$ + ({round(b, 3)})=125(c=125)')

        ax.set_xlabel('x4')
        ax.set_ylabel('x5')
        x = np.arange(-20, 20, 0.05)
        ax.plot(x, np.zeros_like(x), '-.')
        x = np.arange(-100, 100, 0.05)
        ax.plot(np.zeros_like(x), x, '-.')
        ax.legend()
        self.ax = ax
        plt.grid()
        # plt.show()

    def change_columns(self, on_what, from_what):
        for i in range(self.rows):
            self.table[i][on_what], self.table[i][from_what] = self.table[i][from_what], self.table[i][on_what]

    def full_search(self, draw=True):
        alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()
        pars = []
        count_var = self.columns + self.rows - 4
        for i in range(1, count_var + 1):
            for j in range(1, i + 1):
                if i != j:
                    pars.append((f'x{i}', f'x{j}'))

        self.results = {}
        for par in pars:
            for i, var in enumerate(par):
                i += 1
                row, column = self.find_variable(var)  # Координаты первого символа в таблица
                if row == 0:
                    if column != i:
                        self.change_columns(on_what=i, from_what=column)
                else:  # значит переменная не на верхней строке
                    column = i
                    self.change_basic(row=row, column=column)
            letter = alphabet[0]
            alphabet.remove(letter)
            self.check_positive()
            self.results[par] = {'result': self.table[-1][-1], 'Опорное': 'Да' if self.check_positive() else 'Нет'}
            print(par, letter)
            self.print()
            if draw:
                self.make_point(letter)

        print(sorted(sorted(self.results.items(), key=lambda x: x[-1]['result'], reverse=True),
                     key=lambda x: x[-1]['Опорное']))
        if draw:
            plt.show()

    def make_point(self, letter):
        row, column = self.find_variable('x4')
        if row == 0:
            x4 = 0
        else:
            x4 = self.table[row][-1]
        row, column = self.find_variable('x5')
        if row == 0:
            x5 = 0
        else:
            x5 = self.table[row][-1]

        self.ax.annotate(f'{letter}', xy=(x4, x5), xytext=(x4, x5))

    def check_positive(self):
        for i in range(1, self.rows - 1):
            if self.table[i][-1] < 0:
                return False
        return True

    def find_variable(self, name):
        for i in range(1, self.columns):
            if name in self.table[0][i]:
                return 0, i
        for i in range(1, self.rows):
            if name in self.table[i][0]:
                return i, 0


if __name__ == '__main__':
    A = np.array([
        [3, 2, 5, 3, 8],
        [-6, -5, 0, 6, -4],
        [7, 3, 3, 9, 5]
    ])
    c = np.array([[0, 7, -5, 7, -5]])
    b = np.array([[21], [-9], [27]])
    table = SymplexTable(A, b, c)

    table.make_self_basic()
    table.draw_on_nonbasic()
    table.full_search()
