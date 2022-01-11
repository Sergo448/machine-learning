import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Загрузка из Интернера данных, запись их в объект DataFrame и вывод на печать

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print('массив')
print(df.to_string())

# Выборка из DF 100 строк элементов (столбец 4 - название цветков)

y = df.iloc[0:100, 4].values
print("Значение четвертого столбца Y - 100")
print(y)

# Преобразование названий цветков (столбец 4) в одномерный массив (вектор) из -1 и 1

y = np.where(y == 'Iris-setosa', -1, 1)
print("Значение названий цветков в виде -1 и 1, Y - 100")
print(y)

# Выборка из объекта DF массива 100 элементов (столбец 0 и столбец 2),
# загрузка его в массив Х (матрица) и печать

X = df.iloc[0:100, [0, 2]].values
print('Значение X - 100')
print(X)
print("Конец X")

# Формирование параметров хначений для вывода на график
# Первые 50 элементов обучающей выборки (строки 0 - 50, столбцы 0, 1)

plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='щетинистый')

# Следующие 50 элементов обучающей выборки (строки 50 - 100, столбцы 0, 1)

plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')

# Формирование названий осей и вывод графика на экран
plt.xlabel('Длина чашелистика')
plt.ylabel('Длина лепестка')
plt.legend(loc='upper left')
plt.show()


class Perceptron(object):
    """
    Классификатор на основе персептрона.
    Параметры
    eta:float - Темп обучения (между 0.0 и 1.0)
    n_iter:int - Проходы по тренировочному набору данных.
    Атрибуты
    w_: 1-мерный массив - Весовые коэффициенты после подгонки.
    errors_: список - Число случаев ошибочной классификации в каждой эпохе.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta  # темп обучений от 0 до 1
        self.n_iter = n_iter  # количество итераций (уроков)

    '''
    Выполнить подгонку модели под тестировачные данные.
    Параметры
    Х: массив, форма - Х[n_samples, n_features],
    где 
                    n_samples - число образцов,
                    n_features - число признаков,
    у: массив, форма = [n_samples] Целевые зеачения.
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


# Тренировка (j, extybt)


ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Число случаев ошибочной классификации')
plt.show()

# Визуализация границы решений
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


class AdaptiveLinearNeuron(object):
    """
    Классификатор на основе ADALINE (ADAptive Linear NEuron)
    Параметры
    eta: float - Темп обучения между 0.0 и 1.0
    niter: in - Проходы по тренировочному набору данных
    Атрибуты
    w_ : 1- мерный массив - Веса после подгонки.
    errors_: список - Число случаев ошибочной классификации в каждой эпохе.
    """

    def __init__(self, rate=0.01, niter=10):
        self.rate = rate  # темп обучения между 0.0 и 1.0
        self.niter = niter  # проходы по тренировочному набору данных

    def fit(self, X, y):
        """
        Выполнить подгонку модели под тестировачные данные.
        Параметры
        Х - (массив), форма  = [n_samples, n_features] - тренировочные вектора,
        где
                        n_samples - число образцов,
                        n_features - число признаков,
        у - (массив), форма - [n_samples] - целевые значения.
        Возвращает
        self: object
        """

        self.weight = np.zeros(1 + X.shape[1])  # weight - одномерный массив - веса после обучения
        self.cost = []  # cost - список ошибок классификации в кажой эпохе
        for i in range(self.niter):
            output = self.net_input(X)
            errors = y - output
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)
        return self

    """ Рассчитать чистый вход """

    def net_input(self, X):
        # вычисление чистого входного сигнала
        return np.dot(X, self.weight[1:]) + self.weight[0]

    """ Вернуть метку класса после единичного скачка """

    def activation(self, X):
        # вычисление линейной активации
        return self.net_input(X)

    def predict(self, X):
        # Возвращаем метку класса после единичного шага (предсказания)
        return np.where(self.activation(X) >= 0.0, 1, -1)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

#learning rate = 0.01

aln1 = AdaptiveLinearNeuron(0.01, 10).fit(X, y)

ax[0].plot(range(1, len(aln1.cost) + 1), np.log10(aln1.cost), marker='o')

ax[0].set_xlabel("Эпохи")
ax[0].set_ylabel("log(Сумма квадратичных ошибок)")
ax[0].set_title('ADALINE. Темп обучения 0.01')

#learning rate 0.0001

aln2 = AdaptiveLinearNeuron(0.0001, 10).fit(X, y)

ax[1].plot(range(1, len(aln2.cost) + 1), aln2.cost, marker='x')

ax[1].set_xlabel("Эпохи")
ax[1].set_ylabel("log(Сумма квадратичных ошибок)")
ax[1].set_title('ADALINE. Темп обучения 0.0001')

plt.show()

""" Стандартизация обучающей выборки и проведение обечение адаптивного нейрна на ее основе """
# Стандартизируем общую выборку
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Обучение на стандартизированной выборке при rate = 0.01
aln = AdaptiveLinearNeuron(0.01, 10)
aln.fit(X_std, y)

# Строим график зависимости стоимости ошибок от эпох обучения

plt.plot(range(1, len(aln.cost) + 1), aln.cost, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Сумма квадратичных ошибок')
plt.show()

# Строим области принятия решений

plot_decion_regions(X_std, y, classifer=aln)
plt.title("ADALINE (градиентный спуск)")
plt.xlabel('Длина чашелистика [стандартизированная]')
plt.ylabel('Длина лепестка [стандартизированная]')
plt.legend(loc='upper left')
plt.show()

""" Проверка успешности обучения адаптивного персептрона 
    Передаем на вход персептрона значение параметров цветков, отсутвующих в обучающей выборке"""

i1 = [-1.5, -0.75]
# i1 = [0.25, 1.1]
R1 = aln.predict(i1)
print('R1 =', R1)

if R1 == 1:
    print('R1= Вид Iris setosa')
else:
    print('R1 = Вид Iris versicolor')
