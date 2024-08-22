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
    def __init__(self, num_agents=2, is_agent_silent=False):
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
        self.agents = []
        for i in range(self.num_agents):
            if i == 0:
                start_x, start_y = UNIT / 2, UNIT / 2  # Top-left for Agent 1
            elif i == 1:
                start_x, start_y = (WIDTH - 0.5) * UNIT, UNIT / 2  # Top-right for Agent 2
            
            agent = {'id': i, 'image': self.shapes[0], 'coords': [start_x, start_y]}
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
        rectangle = PhotoImage(Image.open("../img/agent.png").resize((65, 65)))
        triangle = PhotoImage(Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("../img/circle.png").resize((65, 65)))
        return rectangle, triangle, circle

    def reset(self):
        self.update()
        time.sleep(0.5)
        
        # Reinitialize agents' positions
        agent_positions = [
            [UNIT / 2, UNIT / 2],  # Top-left for Agent 1
            [(WIDTH - 0.5) * UNIT, UNIT / 2]  # Top-right for Agent 2
        ]
        
        for agent, pos in zip(self.agents, agent_positions):
            x, y = pos
            self.canvas.coords(agent['image_obj'], x, y)
            agent['coords'] = [x, y]

        self.update_grid_colors()
        self.messages = [None] * len(self.agents)

        observations = []
        win_state = False
        for agent in self.agents:
            state = self.coords_to_state(agent['coords'])
            if self.is_agent_silent:
                communication_observation = None
            else:
                communication_observation = None  # Placeholder for communication
            
            observation = [state, win_state, communication_observation]

            # observation = {
            #     'physical_observation': state,
            #     'communication_observation': communication_observation
            # }
            observations.append(observation)
            # observations.append(win_state)


        return observations

    def step(self, actions):
        rewards = []
        dones = []
        wins = []
        next_states = []
        self.render()

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
            state = agent['coords']
            base_action = np.array([0, 0])

            # print (f"check agent {idx}, physical action: {action[0]}, comm action: {action[1]}")
            # print (f"check actions {actions}")

            if self.is_agent_silent:
                if action[0] == 1:  # up
                    if state[1] > UNIT:
                        base_action[1] -= UNIT
                elif action[0] == 2:  # down
                    if state[1] < (HEIGHT - 1) * UNIT:
                        base_action[1] += UNIT
                elif action[0] == 3:  # left
                    if state[0] > UNIT:
                        base_action[0] -= UNIT
                elif action[0] == 4:  # right:
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

            # Move the agent and update its state
            self.canvas.move(agent['image_obj'], base_action[0], base_action[1])
            self.canvas.tag_raise(agent['image_obj'])
            next_state = self.canvas.coords(agent['image_obj'])
            agent['coords'] = next_state

            # Determine reward and check if done
            reward = 0
            done = False
            win = False
            # print(f"target coordinate: {self.get_circle_grid_position()}")
            # print(f"next state: {next_state}")
            if next_state == self.canvas.coords(self.circle):  # Agent hits the target
                reward = 100
                done = True
                win = True
            elif next_state in [self.canvas.coords(self.triangle1), self.canvas.coords(self.triangle2)]:  # Agent hits an obstacle
                reward = -100
                done = True
                win = False
                self.update_grid_colors((255, 0, 0))
            else:
                reward = -1

            # Append reward and done status
            rewards.append(reward)
            dones.append(done)
            wins.append(win)

            # Prepare next state observation
            next_state_obs = self.coords_to_state(next_state)
            # next_state_obs.append(win)

            # Append received message to observation if communication is enabled
            if not self.is_agent_silent:
                # print('agents not silent')
                next_state_comms = []
                for other_agent in self.agents:
                    if other_agent == agent:
                        continue  # Skip the current agent itself

                    # Append other agent's communication message
                    # other_agent_message = f"{agent['id']} will receive message from other agent {other_agent['id']}"
                    # other_agent_message = f"{agent['id']} has receive message from other agent {other_agent['id']}: {actions[other_agent['id']][1]}"
                    other_agent_message = actions[other_agent['id']][1]
                    next_state_comms.append(other_agent_message)

                # next_state_obs += next_state_comms
            # else:
                # print('agents is silent')
            # Append next state observation to list
            next_state_observation = [next_state_obs, win, next_state_comms]

            next_states.append(next_state_observation)

        # Check if all agents are done
        if all(dones) and all(wins):
            self.update_grid_colors((0, 255, 0))

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

    def get_circle_grid_position(self):
        circle_coords = self.canvas.coords(self.circle)
        grid_position = self.coords_to_state(circle_coords)
        return grid_position

    @staticmethod
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb


if __name__ == "__main__":
    env = Env()
    # print("Circle Grid Position:", env.get_circle_grid_position())
    env.mainloop()
