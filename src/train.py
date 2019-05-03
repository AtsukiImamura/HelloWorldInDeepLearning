import sys, os, math
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt

from Net import Net
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

network = Net()

# iters_num = 120
iters_num = 12000
train_size = x_train.shape[0]
print('train_size:'+str(train_size))
print(x_train.shape)

batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size , 1)/2
iter_per_epoch = 100
# print('iter_per_epoch = '+str(iter_per_epoch))

for index in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    network.gradient(x_batch, t_batch)
    loss = network.loss(x_batch, t_batch)

    train_loss_list.append(loss)

    if index % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print('['+str(index).ljust(math.ceil(math.log10(iters_num)))+'] loss: {:.5f}'.format(loss), end = '')
        print('    ', 'train acc: {:.3f}'.format(train_acc), '   test acc: {:.3f}'.format(test_acc))

print(np.arange(0, len(train_acc_list), 1))
print(train_acc_list)
plt.plot(np.arange(0, len(train_acc_list), 1), train_acc_list, label='train acc')
plt.plot(np.arange(0, len(test_acc_list), 1), test_acc_list, linestyle='--', label='test acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()