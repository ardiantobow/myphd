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
    def __init__(self, num_agents=2, is_agent_silent=True):
        super(Env, self).__init__()
        self.action_space = ['stay', 'u', 'd', 'l', 'r', 'send']
        self.n_actions = len(self.action_space)
        self.is_agent_silent = is_agent_silent
        self.title('Multi-Agent Environment')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.grid_colors = [[(255, 255, 255)] * WIDTH for _ in range(HEIGHT)]
        self.texts = []
        self.obstacle_direction = 1  # 1 for moving right, -1 for moving left

        # Multi-agent setup
        self.num_agents = num_agents
        self.agents = []
        self.messages = []
        self.init_agents()
        self.canvas = self._build_canvas()

    def init_agents(self):
        for i in range(self.num_agents):
            agent = {'id': i + 1, 'image': self.shapes[0], 'coords': [UNIT * (i + 1.5), UNIT * (i + 1.5)]}
            self.agents.append(agent)
        self.messages = [None] * len(self.agents)  # Initialize messages to None

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white', height=HEIGHT * UNIT, width=WIDTH * UNIT)
        for r in range(0, HEIGHT * UNIT, UNIT):
            for c in range(0, WIDTH * UNIT, UNIT):
                x0, y0, x1, y1 = c, r, c + UNIT, r + UNIT
                grid_color = self.grid_colors[r // UNIT][c // UNIT]
                canvas.create_rectangle(x0, y0, x1, y1, fill=self.rgb_to_hex(grid_color), outline='black')

        for agent in self.agents:
            agent['image_obj'] = canvas.create_image(agent['coords'][0], agent['coords'][1], image=agent['image'])

        self.triangle1 = canvas.create_image(250, 150, image=self.shapes[1])
        self.triangle2 = canvas.create_image(150, 250, image=self.shapes[1])
        self.circle = canvas.create_image(250, 250, image=self.shapes[2])

        canvas.pack()
        return canvas

    def load_images(self):
        rectangle = PhotoImage(Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("../img/circle.png").resize((65, 65)))
        return rectangle, triangle, circle

    def reset(self):
        self.update()
        time.sleep(0.5)
        for agent in self.agents:
            x, y = agent['coords']
            self.canvas.move(agent['image_obj'], UNIT / 2 - x, UNIT / 2 - y)
            agent['coords'] = [UNIT / 2, UNIT / 2]

        self.render()
        self.update_grid_colors()
        self.messages = [None] * len(self.agents)

        observations = [self.coords_to_state(agent['coords']) for agent in self.agents]
        if not self.is_agent_silent:
            observations = [obs + [None] for obs in observations]

        return observations

    def step(self, actions):
        rewards = []
        dones = []
        next_states = []

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
            state = agent['coords']
            base_action = np.array([0, 0])
            message = None

            if self.is_agent_silent:
                if action == 1:  # up
                    if state[1] > UNIT:
                        base_action[1] -= UNIT
                elif action == 2:  # down
                    if state[1] < (HEIGHT - 1) * UNIT:
                        base_action[1] += UNIT
                elif action == 3:  # left
                    if state[0] > UNIT:
                        base_action[0] -= UNIT
                elif action == 4:  # right:
                    if state[0] < (WIDTH - 1) * UNIT:
                        base_action[0] += UNIT
            else:
                if action[0] == 1:  # up
                    if state[1] > UNIT:
                        base_action[1] -= UNIT
                elif action[0] == 2:  # down
                    if state[1] < (HEIGHT - 1) * UNIT:
                        base_action[1] += UNIT
                elif action[0] == 3:  # left
                    if state[0] > UNIT:
                        base_action[0] -= UNIT
                elif action[0] == 4:  # right
                    if state[0] < (WIDTH - 1) * UNIT:
                        base_action[0] += UNIT
                if action[1] == 'send':  # send message
                    message = f"Message from agent {agent['id']}"

            self.canvas.move(agent['image_obj'], base_action[0], base_action[1])
            self.canvas.tag_raise(agent['image_obj'])
            next_state = self.canvas.coords(agent['image_obj'])
            agent['coords'] = next_state

            reward = 0
            done = False
            if next_state == self.canvas.coords(self.circle):  # Agent hits the target
                reward = 100
                done = True
                self.update_grid_colors((0, 255, 0))
            elif next_state in [self.canvas.coords(self.triangle1), self.canvas.coords(self.triangle2)]:  # Agent hits an obstacle
                reward = -100
                done = True
                self.update_grid_colors((255, 0, 0))
            else:
                reward = -1

            rewards.append(reward)
            dones.append(done)
            next_state_obs = self.coords_to_state(next_state)
            if not self.is_agent_silent:
                next_state_obs += [self.messages[idx]]
                self.messages[idx] = message
            next_states.append(next_state_obs)

        return next_states, rewards, dones

    def render(self):
        time.sleep(0.03)
        self.update()

    def update_grid_colors(self, color=(255, 255, 255)):
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
