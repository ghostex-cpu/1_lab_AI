import math
import matplotlib.pyplot as plt
import torch
import numpy as np

#cos(x + pi/2) + atan(x)
x_train = torch.rand(1000) * 6 * math.pi
y_train = np.cos(x_train + math.pi / 2) + np.arctan(x_train)

x_test = torch.rand(1000) * 6 * math.pi
x_test.numpy().sort()
y_test = np.cos(x_test + math.pi / 2) + np.arctan(x_test)

plt.plot(x_test.numpy(), y_test.numpy())
plt.savefig("mygraph.png")

class Network(torch.nn.Module):

    def __init__(self, hidden_neurons):
        super(Network, self).__init__()
        #1 слой принимает сигнал
        self.fc1 = torch.nn.Linear(1, hidden_neurons)
        #функция сигмоиды для первого слоя
        self.act1 = torch.nn.Sigmoid()
        #3 функция активации для выходного слоя
        self.fc3 = torch.nn.Linear(hidden_neurons, 1)
       


    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc3(x)
        return x


#функция потерь (штрафы для ошибки ИИ)
def loss(pred, target):
    ans = (pred-target)**2
    return ans.mean()

#Функция рассчета всего множества входных данных
#с одновременной отрисовкой 
def predict(net, x, y):
    y_pred = net.forward(x).detach()
    plt.clf()
    plt.plot(x.numpy(), y.numpy(), 'o', c='b', label='Ground truth')
    plt.plot(x.numpy(), y_pred.numpy(), 'o', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')


def lab_1(hidden_neurons, h):
    hidden_list = [2, 3, 5, 10]

        
    x_train = x_train.unsqueeze(1)
    y_train = y_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    for h in hidden_list:

        print(f"\n hidden_neurons = {h}")

        our_network = Network(h)
        optimizer = torch.optim.Adam(our_network.parameters(), 0.01)
        
        for i in range(50000):
            optimizer.zero_grad()
            y_pred = our_network.forward(x_train)
            loss_val = loss(y_pred, y_train)
            loss_val.backward()
            optimizer.step()
        
            if i % 5000 == 0:
                y_pred_test = our_network.forward(x_test)
                print(((y_pred_test - y_test) ** 2).mean() ** 0.5)
                predict(our_network, x_test, y_test)
                plt.savefig(f"mygraph-{i}-h-{h}.png")
        
                    

if __name__ == '__main__':
    hidden_list = [2, 3, 5, 10]

        
    x_train = x_train.unsqueeze(1)
    y_train = y_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    for h in hidden_list:

        print(f"\n hidden_neurons = {h}")

        our_network = Network(h)
        optimizer = torch.optim.Adam(our_network.parameters(), 0.01)
        
        for i in range(50000):
            optimizer.zero_grad()
            y_pred = our_network.forward(x_train)
            loss_val = loss(y_pred, y_train)
            loss_val.backward()
            optimizer.step()
        
            if i % 5000 == 0:
                y_pred_test = our_network.forward(x_test)
                print(((y_pred_test - y_test) ** 2).mean() ** 0.5)
                predict(our_network, x_test, y_test)
                plt.savefig(f"1mygraph-{i}-h-{h}.png")
        
                