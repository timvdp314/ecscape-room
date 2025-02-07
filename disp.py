import pygame
import numpy as np
import logging

import pygame.surface

class EnvRenderer:
    def __init__(self, window : pygame.Surface, window_size, grid_size):
        self.canvas = None
        self.window = window

        self.window_size = window_size
        self.grid_size = grid_size
        self.grid_square_size = self.window_size / self.grid_size

    def draw_object(self, grid_pos, img, scale = (1.0, 1.0)):
        img_data = pygame.image.load("img/" + img)

        if (scale != (1.0, 1.0)):
            img_data = pygame.transform.scale(img_data, np.array(img_data.get_size()) * scale)

        img_size = np.array(img_data.get_size())
        x_y_pos = np.array(grid_pos) * self.grid_square_size + (np.array((self.grid_square_size, self.grid_square_size)) - img_size) // 2

        self.canvas.blit(img_data, x_y_pos)

    def draw_solid_square(self, grid_pos, color: tuple[int, int, int]):
        pygame.draw.rect(self.canvas, color, (grid_pos[0] * self.grid_square_size, grid_pos[1] * self.grid_square_size, self.grid_square_size, self.grid_square_size))

    def draw_gridlines(self):
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                self.canvas,
                0,
                (0, self.grid_square_size * x),
                (self.window_size, self.grid_square_size * x),
                3
            )

            pygame.draw.line(
                self.canvas,
                0,
                (self.grid_square_size * x, 0),
                (self.grid_square_size * x, self.window_size),
                3
            )

    def clear_frame(self):
        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))

    def clear_screen(self):
        self.clear_frame()
        self.render_frame()

    def render_frame(self):
        if (self.canvas is None):
            logging.warning("Canvas is empty, cannot render!")
            return
        
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.display.update()