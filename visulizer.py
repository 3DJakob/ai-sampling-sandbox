import pygame

# create pygame window
# display circles in window from coordinates -1 to 1

def init():
    pygame.init()
    pygame.display.set_caption("PyTorch Visualizer")
    screen = pygame.display.set_mode((800, 800))
    return screen

def draw(screen, data, target):
    screen.fill((255, 255, 255))
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][1]
        if target[i] == 1:
            pygame.draw.circle(screen, (0, 0, 255), (int(x * 400 + 400), int(y * 400 + 400)), 5)
        else:
            pygame.draw.circle(screen, (255, 0, 0), (int(x * 400 + 400), int(y * 400 + 400)), 5)
    pygame.display.update()
