import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 10  # grid height
WIDTH = 10  # grid width


class Env(tk.Tk):
    def __init__(self, num_agents=2, num_obstacles=2, obstacles_random_steps=20, is_agent_silent=False):
        super(Env, self).__init__()
        self.action_space = ['s', 'u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.is_agent_silent = is_agent_silent
        self.title('Multi-Agent Dynamic Environment with Sensory Information')
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
        self.obstacle_directions = [(1, 0), (0, 1)] * (num_obstacles // 2) + [(-1, 0), (0, -1)] * ((num_obstacles + 1) // 2)
        self.first_agent_reached = False
        self.mega_bonus_given = False
        self.win_flag = False
        self.locked = [False] * self.num_agents
        self.win = [False] * self.num_agents  
        self.init_agents()
        self.canvas = self._build_canvas()
        self.next_state_comms = [[] for _ in range(self.num_agents)]

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

        agent_positions = [(agent['coords'][0], agent['coords'][1]) for agent in self.agents]  # Get agents' initial positions

        for i in range(self.num_obstacles):
            if i < len(obstacle_positions):
                pos = obstacle_positions[i]
            else:
                # If more obstacles than default positions, randomly place within the grid without overlapping agents
                while True:
                    x = np.random.randint(0, WIDTH) * UNIT + UNIT / 2
                    y = np.random.randint(0, HEIGHT) * UNIT + UNIT / 2
                    pos = (x, y)
                    if pos not in agent_positions:  # Ensure the obstacle doesn't overlap with agents
                        break

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
        self.win = [False] * self.num_agents
        self.next_state_comms = [[0] for _ in range(self.num_agents)]  # Initialize with zero communication state 

        self.episode_count += 1  # Increment episode counter

        # Move obstacles every 'obstacles_random_steps' episodes
        if self.episode_count % self.obstacles_random_steps == 0:
            self.move_obstacles()

        observations = []
        win_state = False
        for agent in self.agents:
            state = self.coords_to_state(agent['coords'])
            sensory_grid = self.get_sensory_grid(agent['coords'])  # Get sensory grid for agent
            if self.is_agent_silent:
                communication_observation = []
            else:
                communication_observation = []  # Placeholder for communication
            
            observation = [state, win_state, sensory_grid, communication_observation]

            observations.append(observation)

        return observations

    def get_sensory_grid(self, coords):
        x, y = self.coords_to_state(coords)
        sensory_grid = []

        for r in range(y - 1, y + 2):
            row = []
            for c in range(x - 1, x + 2):
                if 0 <= r < HEIGHT and 0 <= c < WIDTH:
                    grid_content = self.get_grid_content(c, r)
                    row.append(grid_content)
                else:
                    row.append(None)  # Outside of bounds
            sensory_grid.append(row)

        return sensory_grid

    def get_grid_content(self, x, y):
        # Check if any agent, obstacle, or the target is in this grid cell
        for agent in self.agents:
            agent_coords = self.coords_to_state(agent['coords'])
            if agent_coords == [x, y]:
                return 'agent'

        for obstacle in self.obstacles:
            obstacle_coords = self.coords_to_state(self.canvas.coords(obstacle))
            if obstacle_coords == [x, y]:
                return 'obstacle'

        target_coords = self.coords_to_state(self.canvas.coords(self.circle))
        if target_coords == [x, y]:
            return 'target'

        return 'empty'

    def highlight_all_sensory_grids(self):
        # Clear previous highlights
        self.update_grid_colors()

        # Highlight each agent's sensory grid in light blue
        for agent in self.agents:
            x, y = self.coords_to_state(agent['coords'])
            for r in range(y - 1, y + 2):
                for c in range(x - 1, x + 2):
                    if 0 <= r < HEIGHT and 0 <= c < WIDTH:
                        self.grid_colors[r][c] = (173, 216, 230)  # Light blue color

        self._update_canvas_colors()

    def move_obstacles(self):
        # Move obstacles in a fixed pattern
        new_positions = set()  # Set to track new positions of obstacles

        for i, obstacle in enumerate(self.obstacles):
            direction = self.obstacle_directions[i % len(self.obstacle_directions)]
            x_move, y_move = direction[0] * UNIT, direction[1] * UNIT

            current_coords = self.canvas.coords(obstacle)
            new_x = current_coords[0] + x_move
            new_y = current_coords[1] + y_move

            # Check for grid boundaries
            if new_x < UNIT / 2 or new_x > (WIDTH - 0.5) * UNIT or new_y < UNIT / 2 or new_y > (HEIGHT - 0.5) * UNIT:
                # Reverse direction if an obstacle reaches the boundary
                self.obstacle_directions[i] = (-direction[0], -direction[1])
                x_move, y_move = -x_move, -y_move
                new_x = current_coords[0] + x_move
                new_y = current_coords[1] + y_move

            # Check if the new position overlaps with any existing obstacles
            if (new_x, new_y) in new_positions:
                # Reverse direction if new position is already occupied
                self.obstacle_directions[i] = (-self.obstacle_directions[i][0], -self.obstacle_directions[i][1])
                x_move, y_move = -x_move, -y_move
                new_x = current_coords[0] + x_move
                new_y = current_coords[1] + y_move

            # Check again for boundaries after reversing direction
            if new_x < UNIT / 2 or new_x > (WIDTH - 0.5) * UNIT or new_y < UNIT / 2 or new_y > (HEIGHT - 0.5) * UNIT:
                # Keep within boundaries after reversing
                new_x = current_coords[0]
                new_y = current_coords[1]

            # Move the obstacle if the new position is not occupied
            if (new_x, new_y) not in new_positions:
                self.canvas.move(obstacle, x_move, y_move)
                new_positions.add((new_x, new_y))

    def step(self, actions):
        # The existing step method has been preserved
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
                next_state_obs = self.coords_to_state(agent['coords'])
                sensory_grid = self.get_sensory_grid(agent['coords'])
                next_states.append([next_state_obs, self.win[idx], sensory_grid, self.next_state_comms[idx]])
                print(f"agent {idx} is locked. Done status: {self.locked[idx]}, win status: {self.win[idx]}")
                continue

            state = agent['coords']
            base_action = np.array([0, 0])
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
                reward_bonus = 100
                done = False
                self.win[idx] = True
                self.locked[idx] = True
                self.update_grid_colors((0, 0, 255))
                print(f"agent {idx} reach the target!")
                
            elif next_state in [self.canvas.coords(obstacle) for obstacle in self.obstacles]:  # Agent hits an obstacle
                reward_bonus = -100
                done = False
                self.win[idx] = False
                self.locked[idx] = True
                self.update_grid_colors((255, 0, 0))
                print(f"agent {idx} hit the obstacle!")
            else:
                reward_bonus = -1
                done = False
                self.win[idx] = False
                self.locked[idx] = False
                self.update_grid_colors((255, 255, 255))
                print(f"agent {idx} is ongoing!")
                
            reward = reward_bonus

            rewards.append(reward)
            dones.append(done)
            wins.append(self.win[idx])

            next_state_obs = self.coords_to_state(next_state)
            sensory_grid = self.get_sensory_grid(next_state)

            if not self.is_agent_silent:
                for other_agent in self.agents:
                    if other_agent == agent:
                        continue

                    other_agent_message = actions[other_agent['id']][1]
                    self.next_state_comms[idx] = other_agent_message if other_agent_message != 0 else 0

            else:
                self.next_state_comms[idx] = 0

            next_state_observation = [next_state_obs, self.win[idx], sensory_grid, self.next_state_comms[idx]]
            next_states.append(next_state_observation)

            agent['coords'] = next_state

            print(f"win status agent {idx} = {self.win[idx]}")

        if agents_reached_target == self.num_agents and not self.mega_bonus_given:
            for i in range(len(rewards)):
                rewards[i] += 0  # Mega bonus
            self.mega_bonus_given = True

        print(f"wins all agent situation in the environment: {wins}")

        if all(wins):
            self.update_grid_colors((0, 255, 0))
            self.win_flag = True

        if all(self.locked):
            dones = [True] * self.num_agents
            self.locked = [False] * self.num_agents

        # Highlight sensory grids for all agents
        self.highlight_all_sensory_grids()

        return next_states, rewards, dones

    def render(self):
        time.sleep(0.005)
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
