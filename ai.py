import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from data import getBatch
from visualizer import Visualizer
from samplers import uniform, mostLoss, leastLoss, distributeLoss, gradientNorm
from api import logNetwork, logRun

visualizer = Visualizer()

n_epochs = 1
batch_size_train = 1024
mini_batch_size_train = 128
batch_size_test = 1024
learning_rate = 0.005
momentum = 0.5
log_interval = 30

# random_seed = 1
random_seed = torch.randint(0, 100000, (1,)).item()
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linearSize = self.getLinearSize()
        print('Linear size: ' + str(self.linearSize))
        self.fc1 = nn.Linear(self.linearSize, 50)
        self.fc2 = nn.Linear(50, 2)
        
    def getLinearSize (self):
      testMat = torch.zeros(1, 2)
      testMat = self.convForward(testMat)
      testMat = testMat.flatten()
      size = testMat.size().numel()
      return size

    def convForward(self, x) -> torch.Tensor:
      return x

    def forward(self, x):
        x = self.convForward(x)
        x = x.view(-1, self.linearSize)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def trainEpoch(self, epoch):
      network.train()

      # global train_loader
      global train_loader

      # get first data sample in enumarate order from train loader
      batch_idx = 0
      NUMBER_OF_BATCHES = 100

      index = 0

      while index < NUMBER_OF_BATCHES:
        [data, target] = getBatch(batch_size_train)
        # [data, target] = uniform(data, target, mini_batch_size_train)
        [data, target] = mostLoss(data, target, mini_batch_size_train, network)
        # [data, target, importance] = gradientNorm(data, target, mini_batch_size_train, network)

        optimizer.zero_grad()
        output = network(data)
        
        targetPred = torch.argmax(output, dim=1)
        visualizer.draw(data, targetPred)

        # wait for 100ms
        time.sleep(0.1)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
          acc = self.test()
          accPlot.append(acc)
          lossPlot.append(loss.item())

          logRun(
            [],
            [],
            accPlot,
            [],
            lossPlot,
            'sandbox',
            6,
            'most loss',
          )
          # plot(accPlot, None)

          # end time
          end = time.time()

          train_losses.append(loss.item())
      
        batch_idx += 1

      

    def test(self):
      network.eval()
      test_loss = 0
      correct = 0
      total = 0
      # NUMBER_OF_BATCHES = int(test_loader.dataset['x'].shape[0] / mini_batch_size_train)
      NUMBER_OF_BATCHES = 20
      with torch.no_grad():
        batchIndex = 0


        while batchIndex < NUMBER_OF_BATCHES:
          [data, target] = getBatch(batch_size_test)


        # for data, target in test_loader:
          output = network(data)
          test_loss += F.nll_loss(output, target, reduction='sum').item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
          total += target.size(0)

          batchIndex = batchIndex + 1


      test_loss /= total
      test_losses.append(test_loss)
      print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
      accTensor = correct / total
      return accTensor.item()

network = Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []

accPlot = []
lossPlot = []
print('Starting training')

# logNetwork(
#   batch_size_train,
#   batch_size_test,
#   'sandbox',
#   learning_rate,
#   'adam',
#   'cross entropy',
#   'custom',
# )


for epoch in range(1, n_epochs + 1):
  network.trainEpoch(epoch)

# nodes = networkTo3dNodes(network, HEIGHT, WIDTH, CHANNELS)
# log3DNodes(nodes, 'camyleon - mini')