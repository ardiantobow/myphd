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
    def __init__(self, num_agents=2, num_obstacles=2, obstacles_random_steps=20, is_agent_silent=False):
        super(Env, self).__init__()
        self.action_space = ['s', 'u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.is_agent_silent = is_agent_silent
        self.title('Multi-Agent Dynamic Environment')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.grid_colors = [[(255, 255, 255)] * WIDTH for _ in range(HEIGHT)]
        self.texts = []
        self.episode_count = 0  # Initialize episode counter

        # Multi-agent setup
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles  # Number of obstacles is now dynamic
        self.obstacles_random_steps = obstacles_random_steps
        self.agents = []
        self.messages = []
        self.obstacles = []  # List to hold obstacle objects
        self.first_agent_reached = False
        self.mega_bonus_given = False
        self.win_flag = False
        self.locked = [False] * self.num_agents
        self.win = [False] * self.num_agents  
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

        # Initialize obstacles at specific locations
        obstacle_positions = [
            (450, 150),  # Coordinates for the first obstacle
            (150, 450),  # Coordinates for the second obstacle
            (150, 350),  # Coordinates for the third obstacle
            (250, 250)   # Coordinates for the fourth obstacle
        ]

        for i in range(self.num_obstacles):
            if i < len(obstacle_positions):
                pos = obstacle_positions[i]
            else:
                # If more obstacles than default positions, randomly place within the grid without overlapping agents
                x = np.random.randint(0, WIDTH) * UNIT + UNIT / 2
                y = np.random.randint(0, HEIGHT) * UNIT + UNIT / 2
                pos = (x, y)
            obstacle = canvas.create_image(pos[0], pos[1], image=self.shapes[1])
            self.obstacles.append(obstacle)

        self.circle = canvas.create_image(450, 450, image=self.shapes[2])

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
        self.locked = [False] * self.num_agents 

        self.episode_count += 1  # Increment episode counter

        # Move obstacles every 20 episodes
        if self.episode_count % self.obstacles_random_steps == 0:
            self.move_obstacles()

        observations = []
        win_state = False
        for agent in self.agents:
            state = self.coords_to_state(agent['coords'])
            if self.is_agent_silent:
                communication_observation = []
            else:
                communication_observation = []  # Placeholder for communication
            
            observation = [state, win_state, communication_observation]

            observations.append(observation)

        return observations

    def move_obstacles(self):
        # Get the current positions of the agents (which are their initial positions after reset)
        initial_agent_positions = [tuple(agent['coords']) for agent in self.agents]
        
        positions = []
        
        while len(positions) < self.num_obstacles:
            x = np.random.randint(0, WIDTH) * UNIT + UNIT / 2
            y = np.random.randint(0, HEIGHT) * UNIT + UNIT / 2
            new_pos = (x, y)
            
            # Ensure no overlap with other obstacles or initial agent positions
            if new_pos not in positions and new_pos not in initial_agent_positions:
                positions.append(new_pos)
        
        # Set the new positions for the obstacles
        for i, obstacle in enumerate(self.obstacles):
            if i < len(positions):
                self.canvas.coords(obstacle, positions[i][0], positions[i][1])

    def step(self, actions):
        rewards = []
        dones = []
        wins = []
        next_states = []
        agents_reached_target = 0
        self.win_flag = False
        
        self.update_grid_colors()
        circle_pos = self.get_circle_grid_position()
        
        reward_position = 0
        reward_bonus = 0
        reward = 0
        done = False

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):

            if self.locked[idx]:  # If agent is locked, skip action processing
                rewards.append(0)
                dones.append(self.locked[idx])
                wins.append(self.win[idx])
                next_states.append([self.coords_to_state(agent['coords']), self.win[idx], "none"])
                print(f"agent {idx} is locked. Done status: {self.locked[idx]}, win status: {self.win[idx]}")
                continue

            state = agent['coords']
            base_action = np.array([0, 0])
            message = None
            physical_action = action[0]

            if physical_action == 0:
                base_action[0] = base_action[0]
                base_action[1] = base_action[1]
            elif physical_action == 1:  # up
                if state[1] > UNIT:
                    base_action[1] -= UNIT
            elif physical_action == 2:  # down
                if state[1] < (HEIGHT - 1) * UNIT:
                    base_action[1] += UNIT
            elif physical_action == 3:  # left
                if state[0] > UNIT:
                    base_action[0] -= UNIT
            elif physical_action == 4:  # right
                if state[0] < (WIDTH - 1) * UNIT:
                    base_action[0] += UNIT

            initial_pos = self.coords_to_state(state)
            initial_distance = abs(initial_pos[0] - circle_pos[0]) + abs(initial_pos[1] - circle_pos[1])

            self.canvas.move(agent['image_obj'], base_action[0], base_action[1])
            self.canvas.tag_raise(agent['image_obj'])
            next_state = self.canvas.coords(agent['image_obj'])
            
            new_pos = self.coords_to_state(next_state)
            new_distance = abs(new_pos[0] - circle_pos[0]) + abs(new_pos[1] - circle_pos[1])

            reward_position = initial_distance - new_distance

            if next_state == self.canvas.coords(self.circle):  # Agent hits the target
                
                agents_reached_target += 1
                # if not self.first_agent_reached:
                #     reward_bonus = 10
                #     self.first_agent_reached = True
                # else:
                #     reward_bonus = 0

                reward_bonus = 50
                done = False
                self.win[idx] = True
                self.locked[idx] = True
                self.update_grid_colors((0, 0, 255))
                
            elif next_state in [self.canvas.coords(obstacle) for obstacle in self.obstacles]:  # Agent hits an obstacle
                reward_bonus = -10
                done = False
                self.win[idx] = False
                self.locked[idx] = True
                self.update_grid_colors((255, 0, 0))
            else:
                reward_bonus = -1
            
            reward = reward_bonus

            rewards.append(reward)
            dones.append(done)
            wins.append(self.win[idx])
            
            next_state_obs = self.coords_to_state(next_state)
            next_state_comms = []
            if not self.is_agent_silent:
                for other_agent in self.agents:
                    if other_agent == agent:
                        continue

                    other_agent_message = actions[other_agent['id']][1]
                    next_state_comms.append(other_agent_message)

            next_state_observation = [next_state_obs, self.win[idx], next_state_comms]

            next_states.append(next_state_observation)

            agent['coords'] = next_state

            print(f"win status agent {idx} = {self.win[idx]}")
        

        if agents_reached_target == self.num_agents and not self.mega_bonus_given:
            for i in range(len(rewards)):
                rewards[i] += 100  # Mega bonus
            self.mega_bonus_given = True
        
        print(f"wins all agent situation in the environment: {wins}")

        if all(wins):
                self.update_grid_colors((0, 255, 0))
                self.win_flag = True
                
        if all(self.locked):
            dones = [True] * self.num_agents
            self.locked = [False] * self.num_agents
        
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
    # Specify the number of obstacles dynamically
    env = Env() 
    env.mainloop()
