# Модуль Net_Fishing

import pybrain3
import pickle
from numpy import ravel
import matplotlib.pylab as plt
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.tools.xml.networkreader import NetworkReader

# Формирование обучающего набора данных

ds = SupervisedDataSet(4, 1)
ds.addSample([2, 3, 80, 1], [5])
ds.addSample([5, 5, 50, 2], [4])
ds.addSample([10, 7, 40, 3], [3])
ds.addSample([15, 9, 20, 4], [2])
ds.addSample([20, 11, 10, 5], [1])

# Формирование структуры нейронной сети

net = buildNetwork(4, 3, 1, bias=True)

# Тренировка (обучение) нейронной сети с визуализацией рабочих этапов тренировки

trainer = BackpropTrainer(net, dataset=ds, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)
trnerr, valerr = trainer.trainUntilConvergence()

plt.plot(trnerr, 'b', valerr, 'r')
plt.show()

# Запись обученной сети в файл MyNet_Fish.txt

fileObject = open('MyNet_Fish.txt', 'wb')
pickle.dump(net, fileObject)
fileObject.close()


"""__________________________________________________________________________________________________________________"""


# Модуль GFshing_Test

fileObject = open('MyNet_Fish.txt', 'rb')
net2 = pickle.load(fileObject)
fileObject.close()

# Хорошие погодные условия

y = net2.activate([2, 3, 80, 1])
print('Y1=', y)

# Средние погодные условия

y = net2.activate([10, 7, 40, 3])
print('Y2=', y)

# Плохие погодные условия

y = net2.activate([20, 11, 10, 5])
print('Y3=', y)