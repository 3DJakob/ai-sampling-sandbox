import pygame
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# create pygame window
# display circles in window from coordinates -1 to 1

class Visualizer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("PyTorch Visualizer")

        x = 100
        y = 45
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)

        self.screen = pygame.display.set_mode((800, 800))
        self.data = None
        self.target = None

        # private variables
        self._edgeIndicies = None

    def draw(self, data, target, maskFunction):
        self.screen.fill((255, 255, 255))
        for i in range(len(data)):
            x = data[i][0]
            y = data[i][1]
            if target[i] == 1:
                pygame.draw.circle(self.screen, (0, 0, 255), (int(x * 400 + 400), int(y * 400 + 400)), 5)
            else:
                pygame.draw.circle(self.screen, (255, 0, 0), (int(x * 400 + 400), int(y * 400 + 400)), 5)
        # self.drawRing()
        self.drawMask(maskFunction)
        pygame.display.update()

    def drawRing(self):
        # draw ring in center with r = 0.62
        pygame.draw.circle(self.screen, (0, 255, 0), (400, 400), 400*0.77, 2)


    def drawMask(self, maskingFunction):
        # maskingFunction is a function that takes a coordinate and returns a boolean
        # draw red where the maskingFunction returns true

        # Masks wont change so we only need to calculate them once
        if self._edgeIndicies is None:
            # create data matrix with all coordinates
            x = torch.linspace(-1, 1, 800)
            y = torch.linspace(-1, 1, 800)
            xx, yy = torch.meshgrid(x, y)
            data = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)    
    
            data = maskingFunction(data)
            # reshape data from 1x800^2 to 800 x 800
            data = data.reshape(800, 800)

            # Define a convolutional filter to detect edges
            edge_filter = torch.tensor([[1, 1, 1],
                                        [1, -8, 1],
                                        [1, 1, 1]], dtype=torch.float)

            # Apply the filter to the mask tensor using convolution
            edges = F.conv2d(data.float().unsqueeze(0).unsqueeze(0), edge_filter.unsqueeze(0).unsqueeze(0), padding=1)

            # Create a new boolean tensor with the detected edges set to True
            edge_mask = (edges > 0).squeeze(0).squeeze(0)
            indices = torch.nonzero(edge_mask)
        
            self._edgeIndicies = indices

        # draw red at indices
        for i in range(len(self._edgeIndicies)):
            x = self._edgeIndicies[i][0]
            y = self._edgeIndicies[i][1]
            self.screen.set_at((x, y), (255, 0, 0))
        
        pygame.display.update()


    def drawPrediction(self, data, prediction):
        self.screen.fill((255, 255, 255))
        for i in range(len(data)):
            x = data[i][0]
            y = data[i][1]
            if prediction[i] == 1:
                pygame.draw.circle(self.screen, (0, 0, 255), (int(x * 400 + 400), int(y * 400 + 400)), 5)
            else:
                pygame.draw.circle(self.screen, (255, 0, 0), (int(x * 400 + 400), int(y * 400 + 400)), 5)
        pygame.display.update()
