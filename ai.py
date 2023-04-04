import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from data import Batcher
from samplers.samplers import Sampler, uniform, mostLoss, leastLoss, gradientNorm
# from samplers.samplers import uniform, mostLoss, leastLoss, distributeLoss, gradientNorm, mostLossEqualClasses
from api import logNetwork, logRun
from varianceReductionCondition import VarianceReductionCondition 
from samplers.pickers import pickCdfSamples, pickOrderedSamples

reductionCondition = VarianceReductionCondition()
batcher = Batcher()
sampler = Sampler()

sampler.setSampler(gradientNorm)
sampler.setPicker(pickOrderedSamples)
RUNNUMBER = 13
RUNNAME = 'gradientNorm'


n_epochs = 1
batch_size_train = 1024
mini_batch_size_train = 128
batch_size_test = 1024
learning_rate = 0.005
momentum = 0.5
log_interval = 10

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
      NUMBER_OF_BATCHES = 30 * 16
      
      currentTrainingTime = 0

      while batch_idx < NUMBER_OF_BATCHES:
        start = time.time()
        [data, target] = batcher.getBatch(batch_size_train)
        # [data, target] = mostLossEqualClasses(data, target, mini_batch_size_train, network)
        # [data, target] = uniform(data, target, mini_batch_size_train)
        # [data, target] = mostLoss(data, target, mini_batch_size_train, network)
        [data, target] = sampler.sample(data, target, mini_batch_size_train, network)

        # if batch_idx > 10:
        #    [data, target] = mostLossEqualClasses(data, target, mini_batch_size_train, network)
        # else:
        #     [data, target] = uniform(data, target, mini_batch_size_train)
        # [data, target] = mostLossEqualClasses(data, target, mini_batch_size_train, network)

        # reductionCondition.update(importance)
        # if reductionCondition.satisfied.item():
        #    print('Variance reduction condition satisfied')



        optimizer.zero_grad()
        output = network(data)
        
        targetPred = torch.argmax(output, dim=1)
        batcher.draw(data, targetPred)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        duration = time.time() - start
        currentTrainingTime += duration

        # wait for 100ms
        # time.sleep(0.1)
        
        if batch_idx % log_interval == 0:
          acc = self.test()
          accPlot.append(acc)
          lossPlot.append(loss.item())
          lastTimestamp = timestampPlot.__len__() > 0 and timestampPlot[-1] or 0
          timestampPlot.append(lastTimestamp + currentTrainingTime)
          # timestampPlot.append(currentTrainingTime)
          currentTrainingTime = 0

          logRun(
            timestampPlot,
            [],
            accPlot,
            [],
            lossPlot,
            'sandbox - 2 circles',
            RUNNUMBER,
            RUNNAME,
          )

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
          [data, target] = batcher.getBatch(batch_size_test)


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
timestampPlot = []
print('Starting training')

# logNetwork(
#   batch_size_train,
#   batch_size_test,
#   'sandbox - 2 circles',
#   learning_rate,
#   'adam',
#   'cross entropy',
#   'custom',
# )


for epoch in range(1, n_epochs + 1):
  print('Epoch: ' + str(epoch))
  network.trainEpoch(epoch)

# nodes = networkTo3dNodes(network, HEIGHT, WIDTH, CHANNELS)
# log3DNodes(nodes, 'camyleon - mini')