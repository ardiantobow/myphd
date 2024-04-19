import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 5  # grid height
WIDTH = 5  # grid width


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('My Environment')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.grid_colors = [[(255, 255, 255)] * WIDTH for _ in range(HEIGHT)]  # Initialize grid colors to white
        self.canvas = self._build_canvas()
        self.texts = []
        self.obstacle_direction = 1  # 1 for moving right, -1 for moving left

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # create grids
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            for c in range(0, WIDTH * UNIT, UNIT):
                x0, y0, x1, y1 = c, r, c + UNIT, r + UNIT
                grid_color = self.grid_colors[r // UNIT][c // UNIT]
                canvas.create_rectangle(x0, y0, x1, y1, fill=self.rgb_to_hex(grid_color), outline='black')

        # add img to canvas
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        self.triangle1 = canvas.create_image(250, 150, image=self.shapes[1])
        self.triangle2 = canvas.create_image(150, 250, image=self.shapes[1])
        self.circle = canvas.create_image(250, 250, image=self.shapes[2])

        # pack all
        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("../img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        self.render()
        # Reset grid colors to white
        self.update_grid_colors()  # Default color is white
        # return observation
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        # move agent
        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        # move rectangle to top level of canvas
        self.canvas.tag_raise(self.rectangle)
        next_state = self.canvas.coords(self.rectangle)

        # Check for collisions with obstacles or target
        reward = 0
        done = False
        if next_state == self.canvas.coords(self.circle):  # Agent hits the target
            reward = 100
            done = True
            self.update_grid_colors((0, 255, 0))  # Green color
        elif next_state in [self.canvas.coords(self.triangle1), self.canvas.coords(self.triangle2)]:  # Agent hits an obstacle
            reward = -100
            done = True
            self.update_grid_colors((255, 0, 0))  # Red color
        else:
            reward -= 1

        # Move the first obstacle left and right
        triangle1_state = self.canvas.coords(self.triangle1)
        if triangle1_state[0] <= UNIT or triangle1_state[0] >= (WIDTH - 1) * UNIT:
            self.obstacle_direction *= -1
        self.canvas.move(self.triangle1, self.obstacle_direction * UNIT, 0)

        return self.coords_to_state(next_state), reward, done

    def render(self):
        time.sleep(0.03)
        self.update()

    def update_grid_colors(self, color=(255, 255, 255)):  # Default color is white
        for r in range(HEIGHT):
            for c in range(WIDTH):
                self.grid_colors[r][c] = color
        self._update_canvas_colors()

    def _update_canvas_colors(self):
        for r in range(HEIGHT):
            for c in range(WIDTH):
                grid_color = self.grid_colors[r][c]
                rect_id = (r * WIDTH) + c + 1
                self.canvas.itemconfig(rect_id, fill=self.rgb_to_hex(grid_color))

    def coords_to_state(self, coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    @staticmethod
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb


if __name__ == "__main__":
    env = Env()
    env.mainloop()
