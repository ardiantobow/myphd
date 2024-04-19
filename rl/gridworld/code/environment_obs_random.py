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
        self.canvas = self._build_canvas()
        self.texts = []
        self.obstacle_direction = 1  # 1 for moving right, -1 for moving left
        self.original_colors = []  # Store original colors of the grid

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
                x0, y0, x1, y1 = c, r, c + UNIT, r + UNIT
                rectangle = canvas.create_rectangle(x0, y0, x1, y1, fill='white', width=1)
                self.original_colors.append('white')  # Store original color
                canvas.tag_lower(rectangle)
                canvas.create_line(x0, y0, x1, y0, x1, y1, x0, y1, x0, y0)
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

    def text_value(self, row, col, contents, action, font='Helvetica', size=10,
                   style='normal', anchor="nw"):
        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 42, 5
        else:
            origin_x, origin_y = 42, 77

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for x in range(HEIGHT):
            for y in range(WIDTH):
                for action in range(0, 4):
                    state = [x, y]
                    if str(state) in q_table.keys():
                        temp = q_table[str(state)][action]
                        self.text_value(y, x, round(temp, 2), action)

    def coords_to_state(self, coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        # Reset agent position
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        agent_pos = self.coords_to_state(self.canvas.coords(self.rectangle))

        # Reset obstacle and target positions
        obs_x, obs_y = self.canvas.coords(self.triangle1)
        target_x, target_y = self.canvas.coords(self.circle)

        # Move obstacle if it's in the same row as the agent
        if agent_pos[1] == int((obs_y - 50) / UNIT):
            new_y = (agent_pos[1] + 1) * UNIT + 50
            self.canvas.move(self.triangle1, 0, new_y - obs_y)

        # Reset colors of grid
        self.reset_grid_color()

        return agent_pos

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

        # reward function
        reward, done = self.get_reward(next_state)

        next_state = self.coords_to_state(next_state)

        return next_state, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()

    def get_reward(self, next_state):
        if next_state == self.canvas.coords(self.circle):
            self.change_grid_color('green')  # Change grid color to green
            return 100, True
        elif next_state in [self.canvas.coords(self.triangle1), self.canvas.coords(self.triangle2)]:
            self.change_grid_color('red')  # Change grid color to red
            return -100, True
        else:
            return -1, False

    def change_grid_color(self, color):
        for i, rectangle in enumerate(self.canvas.find_all()):
            self.canvas.itemconfig(rectangle, fill=color)
            self.original_colors[i] = color

    def reset_grid_color(self):
        for i, rectangle in enumerate(self.canvas.find_all()):
            self.canvas.itemconfig(rectangle, fill=self.original_colors[i])


if __name__ == "__main__":
    env = Env()
    env.mainloop()
