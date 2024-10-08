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
        self.action_space = ['s', 'u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.is_agent_silent = is_agent_silent
        self.title('Multi-Agent Environment')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.grid_colors = [[(255, 255, 255)] * WIDTH for _ in range(HEIGHT)]
        self.texts = []
        self.episode_count = 0  # Initialize episode counter

        # Multi-agent setup
        self.num_agents = num_agents
        self.agents = []
        self.messages = []
        self.first_agent_reached = False
        self.mega_bonus_given = False
        self.locked = [False] * self.num_agents  # Track locked agents
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
        self.first_agent_reached = False
        self.mega_bonus_given = False
        self.locked = [False] * self.num_agents  # Reset locked status

        self.episode_count += 1  # Increment episode counter

        # Move obstacles every 100 episodes
        if self.episode_count % 100 == 0:
            self.move_obstacles()

        observations = []
        win_state = False
        for agent in self.agents:
            state = self.coords_to_state(agent['coords'])
            if self.is_agent_silent:
                communication_observation = None
            else:
                communication_observation = None  # Placeholder for communication
            
            observation = [state, win_state, communication_observation]

            observations.append(observation)

        return observations

    def move_obstacles(self):
        # Generate random positions for the obstacles
        positions = []
        while len(positions) < 2:
            x = np.random.randint(0, WIDTH) * UNIT + UNIT / 2
            y = np.random.randint(0, HEIGHT) * UNIT + UNIT / 2
            new_pos = (x, y)
            if new_pos not in positions:  # Ensure no overlap
                positions.append(new_pos)

        # Set the new positions for the obstacles
        self.canvas.coords(self.triangle1, positions[0][0], positions[0][1])
        self.canvas.coords(self.triangle2, positions[1][0], positions[1][1])

    def step(self, actions):
        rewards = []
        dones = []
        wins = []
        next_states = []
        
        self.update_grid_colors()
        circle_pos = self.get_circle_grid_position()

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
            if self.locked[idx]:  # If agent is locked, skip action processing
                rewards.append(0)
                dones.append(self.locked[idx])
                wins.append(False)
                next_states.append([self.coords_to_state(agent['coords']), False, None])
                continue

            state = agent['coords']
            base_action = np.array([0, 0])
            physical_action = action[0]

            if physical_action == 0:  # stay
                base_action[0] = base_action[0]
                base_action[1] = base_action[1]
            elif physical_action == 1:  # up
                if state[1] > UNIT:  # Prevent moving out of the top boundary
                    base_action[1] -= UNIT
            elif physical_action == 2:  # down
                if state[1] < (HEIGHT - 1) * UNIT:  # Prevent moving out of the bottom boundary
                    base_action[1] += UNIT
            elif physical_action == 3:  # left
                if state[0] > UNIT:  # Prevent moving out of the left boundary
                    base_action[0] -= UNIT
            elif physical_action == 4:  # right
                if state[0] < (WIDTH - 1) * UNIT:  # Prevent moving out of the right boundary
                    base_action[0] += UNIT

            self.canvas.move(agent['image_obj'], base_action[0], base_action[1])
            self.canvas.tag_raise(agent['image_obj'])
            next_state = self.canvas.coords(agent['image_obj'])
            
            new_pos = self.coords_to_state(next_state)

            if new_pos == self.coords_to_state(self.canvas.coords(self.circle)):  # Agent reaches the target
                rewards.append(100)
                dones.append(True)
                wins.append(True)
                self.locked[idx] = True  # Lock the agent
            elif new_pos in [self.coords_to_state(self.canvas.coords(self.triangle1)), self.coords_to_state(self.canvas.coords(self.triangle2))]:  # Agent hits an obstacle
                rewards.append(-10)
                dones.append(True)
                wins.append(False)
                self.locked[idx] = True  # Lock the agent
            else:
                rewards.append(-1)
                dones.append(False)
                wins.append(False)

            next_states.append([new_pos, wins[-1], None])  # Updated next_state includes win flag

        # If all agents reach the target, give mega bonus
        if all(wins) and not self.mega_bonus_given:
            for i in range(len(rewards)):
                rewards[i] += 1000  # Mega bonus
            self.mega_bonus_given = True
            self.update_grid_colors((0, 255, 0))

        # End the episode if all agents are locked
        if all(self.locked):
            dones = [True] * self.num_agents

        return next_states, rewards, dones

    def render(self):
        time.sleep(0.05)
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
    env.mainloop()
