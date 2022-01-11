# Модуль Net1
import matplotlib.pylab as plt
import numpy as np


# Функция активации: f(x) = 1 / (1 + e^(-x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Создание класса Нейрон

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforword(self, inputs):  # Сумматор
        total = np.dot(self.weights, inputs) + self.bias  # Суммируем выходы
        return sigmoid(total)  # Обращение к функции активации


# Нейронная сеть из 3х слоев

class OurNeuralNetwork:

    def __init__(self):
        weights = np.array([0, 1])  # Веса (одинаковы для вчех нейронов)
        bias = 0

        # Формируем сеть из 3х нейронов
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforword(x)  # Создаем выход Y1 из нейрона h1
        out_h2 = self.h2.feedforword(x)  # Создаем выход Y2 из нейрона h2
        out_o1 = self.o1.feedforword(np.array([out_h1, out_h2]))  # Создаем выход Y из нейрона o1

        return out_o1


network = OurNeuralNetwork()  # Создаем объект СЕТЬ из класса ....
x = np.array([2, 3])  # Формируем входные параметры для сети X1 = 2, X2 = 3
print("Y =", network.feedforward(x))  # Передаем входы в сеть и получаем результат
