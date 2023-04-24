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

sampler.setSampler(uniform)
sampler.setPicker(pickOrderedSamples)


# Variables to be set by the user
NETWORKNAME = 'sandbox - 2 circles results2'
RUNNUMBER = 41
RUNNAME = 'gradient norm 0.28 threshold'
TIMELIMIT = 30
SAMPLINGTHRESHOLD = 0.28
IMPORTANCESAMPLER = gradientNorm
NUMBEROFRUNS = 1

batch_size_train = 1024
mini_batch_size_train = 128
batch_size_test = 1024
learning_rate = 0.005
momentum = 0.5
log_interval = 10
api_interval = 100

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
        self.currentTrainingTime = 0
        self.initialLoss = 0

        # Plotting
        self.lossPlot = []
        self.accPlot = []
        self.timestampPlot = []
        
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

      while self.currentTrainingTime < TIMELIMIT:
        if batch_idx % log_interval == 0:
          acc, loss = self.test()
          self.accPlot.append(acc)
          self.lossPlot.append(loss)
          self.timestampPlot.append(self.currentTrainingTime)

        if batch_idx % api_interval == 0:
          logRun(
            self.timestampPlot,
            [],
            self.accPlot,
            [],
            self.lossPlot,
            NETWORKNAME,
            RUNNUMBER,
            RUNNAME,
          )

        # if batch_idx == 80:
        #   # turn on importance sampling
        #   sampler.setSampler(gradientNorm)

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

        trainLoss = F.cross_entropy(output, target)
        trainLoss.backward()
        optimizer.step()

        self.currentTrainingTime += time.time() - start

        # wait for 100ms
        # time.sleep(0.1)
        
      
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
      if self.initialLoss == 0:
        self.initialLoss = test_loss

      if self.initialLoss * SAMPLINGTHRESHOLD > test_loss:
        print('Sampling threshold reached')
        sampler.setSampler(IMPORTANCESAMPLER)
      
      print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
      accTensor = correct / total
      return accTensor.item(), test_loss
    
    def reset(self):
      self.currentTrainingTime = 0
      # reset weights
      self.fc1.reset_parameters()
      self.fc2.reset_parameters()

      self.accPlot = []
      self.lossPlot = []
      self.timestampPlot = []
      
      global RUNNUMBER
      RUNNUMBER = RUNNUMBER + 1
      

network = Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []

print('Starting training')

# logNetwork(
#   batch_size_train,
#   batch_size_test,
#   NETWORKNAME,
#   learning_rate,
#   'adam',
#   'cross entropy',
#   'custom',
# )


for epoch in range(1, NUMBEROFRUNS + 1):
  print('Epoch: ' + str(epoch))
  network.trainEpoch(epoch)
  network.reset()

# nodes = networkTo3dNodes(network, HEIGHT, WIDTH, CHANNELS)
# log3DNodes(nodes, 'camyleon - mini')