import pygame
import os

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

    def draw(self, data, target):
        self.screen.fill((255, 255, 255))
        for i in range(len(data)):
            x = data[i][0]
            y = data[i][1]
            if target[i] == 1:
                pygame.draw.circle(self.screen, (0, 0, 255), (int(x * 400 + 400), int(y * 400 + 400)), 5)
            else:
                pygame.draw.circle(self.screen, (255, 0, 0), (int(x * 400 + 400), int(y * 400 + 400)), 5)
        self.drawRing()
        pygame.display.update()


    def drawRing(self):
        # draw ring in center with r = 0.62
        pygame.draw.circle(self.screen, (0, 255, 0), (400, 400), 400*0.77, 2)


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
