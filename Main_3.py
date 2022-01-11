import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta  # темп обучений от 0 до 1
        self.n_iter = n_iter  # количество итераций (уроков)

    '''
    Выполнить подгонку модели под тестировачные данные.
    Параметры
    Х - тренировочные данные: массив, размерность - Х[n_samples, n_features],
    где 
                    n_samples - число образцов,
                    n_features - число признаков,
    у - Целевые значения: массив, размерность - y[n_samples]
    Возвращает
    self: object
    '''

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])  # w_ - одномерный массив - веса после обучения
        self.errors_ = []  # errors_ - список ошибок классификации в кажой эпохе
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    """ Рассчитать чистый вход """

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    """ Вернуть метку класса после единичного скачка """

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# Загрузка из Интернера данных, запись их в объект DataFrame и вывод на печать

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print('Данные об Ирисах')
print(df.to_string())
df.to_csv('.Iris.csv', sep='\t', encoding='utf-8')

# Выборка из DF 100 строк (столбец 0 и столбец 2), загрузка их в массив х

X = df.iloc[0:100, [0, 2]].values
print("Значение Х - 100")
print(X)

# Выборка из DF 100 строк (столбец 4 - название цветков) и загрузка их в массив Y

y = df.iloc[0: 100, 4].values

# Преобразование названий цветков (столбец 4) в массив числе -1 и 1

y = np.where(y == 'Iris-setosa', -1, 1)
print("Значение названий цветков в виде -1 и 1, Y - 100")
print(y)

# Первые 50 элементов обучающей выборки (строки 0 - 50, столбцы 0, 1)

plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='щетинистый')

# Следующие 50 элементов обучающей выборки (строки 50 - 100, столбцы 0, 1)

plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')

# Формирование названий осей и вывод графика на экран
plt.xlabel('Длина чашелистика')
plt.ylabel('Длина лепестка')
plt.legend(loc='upper left')
plt.show()

# Тренировка (j, extybt)

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Число случаев ошибочной классификации')
plt.show()

# Проверка
i1 = [5.5, 1.6]
i2 = [6.4, 4.5]
R1 = ppn.predict(i1)
R2 = ppn.predict(i2)
print('R1=', R1, 'R2=', R2)

if R1 == 1:
    print('R1 = Вид Iris setosa')
else:
    print('R1 = Вид Iris versicolor')

# Визуализация разрешительной границы
from matplotlib.colors import ListedColormap


def plot_decion_regions(X, y, classifer, resolution=0.02):
    # Настроить генератор маркеров и палитру
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Вывести поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifer.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Показать образцы классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


# Нарисовать картинку

plot_decion_regions(X, y, classifer=ppn)
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Длина лепестка, см')
plt.legend(loc='upper left')
plt.show()
