# Модуль PyBr_Trener_Ful
import pybrain3
import matplotlib.pylab as plt
from numpy import ravel
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.tools.xml.networkreader import NetworkReader
import pickle

net = buildNetwork(2, 3, 1)

y = net.activate([2, 1])
print('Y=', y)

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))
print(ds)


trainer = BackpropTrainer(net)
trnerr, valerr = trainer.trainUntilConvergence(dataset=ds, maxEpochs=100)

plt.plot(trnerr, 'b', valerr, 'r')
plt.show()

# Распечатка результатов обучения
for mod in net.modules:
    print("Module:", mod.name)
    if mod.paramdim > 0:
        print('--parametrs:', mod.params)
    for conn in net.connections[mod]:
        print('-connection to', conn.outmod.name)
        if conn.paramdim > 0:
            print('-parameters:', conn.params)

    if hasattr(net, 'recurrentConns'):
        print("Recurrent connections")
        for conn in net.recurrentConns:
            print('-', conn.inmod.name, 'to', conn.outmod.name)
            if conn.paramdim > 0:
                print('- parameters', conn.params)

# Проверка работы сети после обучения
y = net.activate([1, 1])
print('Y1 =', y)

# Запись сети в файл txt
fileObject = open('MyNet.txt', 'wb')
pickle.dump(net, fileObject)
fileObject.close()

# Чтение сети из файла txt
fileObject = open('MyNet.txt', 'rb')
net2 = pickle.load(fileObject)

# Проверка работы загруженной из файла сети
y = net2.activate([1, 1])
print('Y2=', y)

""" Запись обученной сети в формате XML"""

MyFile = 'MyNet1.xml'
NetworkWriter.writeToFile(net, MyFile)
NetworkWriter.writeToFile(net, MyFile)
net = NetworkReader.readFrom(MyFile)
