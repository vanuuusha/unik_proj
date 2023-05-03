import numpy as np

class SymplexTable:
    def __init__(self, A, b, c):
        first_line = [' '] + [f'-x{i}' for i in range(1, A.shape[1] + 1)] + ['b']
        last_line = ['L'] + [-i for i in c[0]] + [0]
        self.table = [first_line]
        for i in range(A.shape[0]):
            if b[i, 0] > 0:
                new_line = [f'y{i + 1}'] + [A[i, k] for k in range(A.shape[1])] + [b[i, 0]]
            else:
                new_line = [f'y{i + 1}'] + [-A[i, k] for k in range(A.shape[1])] + [-b[i, 0]]
            self.table.append(new_line)

        self.table.append(last_line)
        self.rows = len(self.table)
        self.columns = len(self.table[0])
        print('Start table')
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

    def find_permissive_column(self):
        minn = 0
        permissive_column = -1
        for j in range(1, self.columns - 1):
            if self.table[-1][j] <= minn:
                minn = self.table[-1][j]
                permissive_column = j
        return permissive_column

    def find_permissive_row(self, permissive_column, y):
        cote = []
        for i in range(1, self.rows):
            if 'L' not in self.table[i][0] and self.table[i][permissive_column] > 0:
                cote.append((self.table[i][-1]/self.table[i][permissive_column], i))
        cote.sort(key=lambda x: x[0])
        if cote:
            return cote[0][1]
        return -1

    def next_symplex_table(self):
        permissive_column = self.find_permissive_column()
        if permissive_column == -1:
            print('Stop')
            return -1
        permissive_row = self.find_permissive_row(permissive_column, y=True)
        if permissive_row == -1:
            print('Stop')
            return -1
        self.change_basic(permissive_row, permissive_column)
        print(f'after permessive elem ({permissive_row + 1}, {permissive_column + 1})')
        self.print()

    def support_task(self):
        L1 = []
        for j in range(1, self.columns - 1):
            temp = 0
            for i in range(1, self.rows - 1):
                temp += -self.table[i][j]
            L1.append(temp)
        self.table.append(['L1'] + [i for i in L1] + [sum([-self.table[i][-1] for i in range(1, self.rows)])])
        self.rows += 1
        print('table with L1')
        self.print()

    def make_self_basic(self):
        self.support_task()
        while self.y_in_basic():
            self.next_symplex_table()
            self.delete_y_column()
        if self.table[-1][-1] == 0:
            print('Successfully brought to own basis')
        else:
            print('Something went wrong =(')
        self.table.pop(-1)
        self.rows -= 1
        print('simplex table in self basic')
        self.print()

    def symplex_method(self):
        while True:
            a = self.next_symplex_table()
            if a == -1:
                break


    def delete_y_column(self):
        for j in range(1, self.columns):
            if 'y' in self.table[0][j]:
                self.delete_column(j)
                break

    def delete_column(self, column_id):
        for i in range(self.rows):
            self.table[i].pop(column_id)
        self.columns -= 1

    def y_in_basic(self):
        for i in range(1, self.rows):
            if 'y' in self.table[i][0]:
                return True
        return False


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
    table.symplex_method()
