import time
import numpy as np
import tkinter as tk

np.random.seed(1)
# self.UNIT = 50  # pixels
# HEIGHT = 20  # grid self.HEIGHT
# WIDTH = 20  # grid self.WIDTH


class Env(tk.Tk):
    def __init__(self, num_agents=2, num_obstacles=2, obstacles_random_steps=20, is_agent_silent=False, sensory_size=3, gpixels=50, gheight=20, gwidth=20, is_sensor_active=True):
        super(Env, self).__init__()
        self.UNIT = gpixels
        self.HEIGHT = gheight
        self.WIDTH = gwidth
        self.action_space = ['s', 'u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.is_agent_silent = is_agent_silent
        self.title('Multi-Agent Dynamic Environment with Sensory Information')
        self.geometry('{0}x{1}'.format(self.HEIGHT * self.UNIT, self.HEIGHT * self.UNIT))
        self.grid_colors = [[(255, 255, 255)] * self.WIDTH for _ in range(self.HEIGHT)]
        self.texts = []
        self.episode_count = 0  # Initialize episode counter
        self.sensory_size = sensory_size
        self.sensory_grid_on = is_sensor_active

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

        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # Ensure cleanup when closing window

    def init_agents(self):
        self.agents = []
        for i in range(self.num_agents):
            if i == 0:
                start_x, start_y = self.UNIT / 2, self.UNIT / 2  # Top-left for Agent 1
            elif i == 1:
                start_x, start_y = (self.WIDTH - 0.5) * self.UNIT, self.UNIT / 2  # Top-right for Agent 2

            agent = {'id': i, 'coords': [start_x, start_y]}
            self.agents.append(agent)
        self.messages = [None] * len(self.agents)  # Initialize messages to None

    def _build_canvas(self):
        # Calculate the center of the environment
        center_x = (self.WIDTH // 2) * self.UNIT + self.UNIT / 2
        center_y = (self.HEIGHT // 2) * self.UNIT + self.UNIT / 2
        
        canvas = tk.Canvas(self, bg='white', height=self.HEIGHT * self.UNIT, width=self.WIDTH * self.UNIT)
        for r in range(0, self.HEIGHT * self.UNIT, self.UNIT):
            for c in range(0, self.WIDTH * self.UNIT, self.UNIT):
                x0, y0, x1, y1 = c, r, c + self.UNIT, r + self.UNIT
                grid_color = self.grid_colors[r // self.UNIT][c // self.UNIT]
                canvas.create_rectangle(x0, y0, x1, y1, fill=self.rgb_to_hex(grid_color), outline='black')

        for agent in self.agents:
            agent_center_x = agent['coords'][0]
            agent_center_y = agent['coords'][1]
            # Correct the agent's position to always be centered in its grid cell
            agent['image_obj'] = canvas.create_oval(agent_center_x - self.UNIT / 4, agent_center_y - self.UNIT / 4,
                                                    agent_center_x + self.UNIT / 4, agent_center_y + self.UNIT / 4,
                                                    fill='blue', outline='black')

        # Initialize obstacles at specific locations
        obstacle_positions = [
            (4.5 * self.UNIT, 1.5 * self.UNIT),  # Coordinates for the first obstacle
            (1.5 * self.UNIT, 4.5 * self.UNIT),  # Coordinates for the second obstacle
            (1.5 * self.UNIT, 3.5 * self.UNIT),  # Coordinates for the third obstacle
            (2.5 * self.UNIT, 2.5 * self.UNIT)   # Coordinates for the fourth obstacle
        ]

        agent_positions = [(agent['coords'][0], agent['coords'][1]) for agent in self.agents]

        for i in range(self.num_obstacles):
            if i < len(obstacle_positions):
                pos = obstacle_positions[i]
            else:
                # Randomly place obstacles within the grid without overlapping agents
                while True:
                    x = np.random.randint(0, self.WIDTH) * self.UNIT + self.UNIT / 2
                    y = np.random.randint(0, self.HEIGHT) * self.UNIT + self.UNIT / 2
                    pos = (x, y)
                    if pos not in agent_positions:  # Ensure no overlap with agents
                        break

            # Properly center obstacles within their grid cells
            obstacle = canvas.create_rectangle(pos[0] - self.UNIT / 4, pos[1] - self.UNIT / 4,
                                            pos[0] + self.UNIT / 4, pos[1] + self.UNIT / 4,
                                            fill='red', outline='black')
            self.obstacles.append(obstacle)


        # Properly center the target within its grid cell
        self.circle = canvas.create_oval(center_x - self.UNIT / 4, center_y - self.UNIT / 4,
                                 center_x + self.UNIT / 4, center_y + self.UNIT / 4,
                                 fill='green', outline='black')

        canvas.pack()
        return canvas

    def reset(self):
        self.update()
        time.sleep(0.5)

        agent_positions = [
            [self.UNIT / 2, self.UNIT / 2],  # Top-left for Agent 1
            [(self.WIDTH - 0.5) * self.UNIT, self.UNIT / 2]  # Top-right for Agent 2
        ]

        for agent, pos in zip(self.agents, agent_positions):
            x, y = pos
            self.canvas.coords(agent['image_obj'], x - self.UNIT / 4, y - self.UNIT / 4, x + self.UNIT / 4, y + self.UNIT / 4)
            agent['coords'] = [x, y]

        self.update_grid_colors()
        self.messages = [None] * len(self.agents)
        self.first_agent_reached = False
        self.mega_bonus_given = False
        self.locked = [False] * self.num_agents
        self.win = [False] * self.num_agents
        self.next_state_comms = [[0] for _ in range(self.num_agents)]
        self.episode_count += 1

        if self.episode_count % self.obstacles_random_steps == 0:
            self.move_obstacles()

        observations = []
        for agent in self.agents:
            state = self.coords_to_state(agent['coords'])
            sensory_grid = self.get_sensory_grid(agent['coords'])
            communication_observation = [] if self.is_agent_silent else []

            observation = [state, False, sensory_grid, communication_observation]
            # observation = [state, False, communication_observation]
            observations.append(observation)

        return observations

    def step(self, actions):
        rewards = []
        dones = []
        next_states = []

        self.update_grid_colors()

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
            if self.locked[idx]:
                rewards.append(0)
                dones.append(True)
                next_state_obs = self.coords_to_state(agent['coords'])
                sensory_grid = self.get_sensory_grid(agent['coords'])
                next_states.append([next_state_obs, self.win[idx], sensory_grid, self.next_state_comms[idx]])
                # next_states.append([next_state_obs, self.win[idx], self.next_state_comms[idx]])
                print(f"Agent {idx} is locked. Done status: {self.locked[idx]}, win status: {self.win[idx]}")
                continue

            state = agent['coords']
            base_action = np.array([0, 0])
            physical_action = action[0]

            if physical_action == 1:  # up
                if state[1] > self.UNIT:
                    base_action[1] -= self.UNIT
            elif physical_action == 2:  # down
                if state[1] < (self.HEIGHT - 1) * self.UNIT:
                    base_action[1] += self.UNIT
            elif physical_action == 3:  # left
                if state[0] > self.UNIT:
                    base_action[0] -= self.UNIT
            elif physical_action == 4:  # right
                if state[0] < (self.WIDTH - 1) * self.UNIT:
                    base_action[0] += self.UNIT

            new_coords = [state[0] + base_action[0], state[1] + base_action[1]]
            self.canvas.coords(agent['image_obj'], new_coords[0] - self.UNIT / 4, new_coords[1] - self.UNIT / 4,
                               new_coords[0] + self.UNIT / 4, new_coords[1] + self.UNIT / 4)
            next_state = self.canvas.coords(agent['image_obj'])

            target_coords = self.canvas.coords(self.circle)

            if next_state == target_coords:  # Agent hits the target
                rewards.append(100)
                self.win[idx] = True
                self.locked[idx] = True
                dones.append(True)
                print(f"Agent {idx} reached the target! Next state: {next_state}, Target: {target_coords}")
            elif next_state in [self.canvas.coords(obstacle) for obstacle in self.obstacles]:  # Agent hits an obstacle
                rewards.append(-100)
                self.locked[idx] = True
                dones.append(True)
                print(f"Agent {idx} hit an obstacle! Next state: {next_state}")
            else:
                rewards.append(-1)
                dones.append(False)

            next_state_obs = self.coords_to_state(next_state)
            sensory_grid = self.get_sensory_grid(new_coords)
            

            if not self.is_agent_silent:
                for other_agent in self.agents:
                    if other_agent == agent:
                        continue

                    other_agent_message = actions[other_agent['id']][1]
                    self.next_state_comms[idx] = other_agent_message if other_agent_message !=0 else 0

            else:
                self.next_state_comms[idx] = 0

            next_state_observation = [next_state_obs, self.win[idx], sensory_grid, self.next_state_comms[idx]]
            # next_state_observation = [next_state_obs, self.win[idx], self.next_state_comms[idx]]
            next_states.append(next_state_observation)


            # Update the agent's position in terms of coordinates
            agent['coords'] = new_coords

        # Highlight sensory grids for all agents (including locked ones)
        if self.sensory_grid_on:
            self.highlight_all_sensory_grids()


        return next_states, rewards, dones

    def highlight_all_sensory_grids(self):
        # Clear previous highlights
        self.update_grid_colors()

        if not self.sensory_grid_on:
            return  # Skip highlighting if the sensory grid is turned off

        half_size = self.sensory_size // 2

        # Highlight each agent's sensory grid
        for agent in self.agents:
            x, y = self.coords_to_state(agent['coords'])
            if self.locked[agent['id']] and self.win[agent['id']]:
                color = (144, 238, 144)  # Light green color (agent hit target)
            elif self.locked[agent['id']]:
                color = (255, 182, 193)  # Light red color (agent hit obstacle)
            else:
                color = (173, 216, 230)  # Light blue color (default)

            for r in range(y - half_size, y + half_size + 1):
                for c in range(x - half_size, x + half_size + 1):
                    if 0 <= r < self.HEIGHT and 0 <= c < self.WIDTH:
                        self.grid_colors[r][c] = color

        self._update_canvas_colors()




    def coords_to_state(self, coords):
        # Convert canvas coordinates to grid indices, keeping the agent in the center
        x = int((coords[0]) // self.UNIT)
        y = int((coords[1]) // self.UNIT)
        return [x, y]

    def get_sensory_grid(self, coords):
        if not self.sensory_grid_on:
            return None  # No sensory grid data if the sensory grid is turned off

        x, y = self.coords_to_state(coords)
        half_size = self.sensory_size // 2
        sensory_grid = []

        for r in range(y - half_size, y + half_size + 1):
            row = []
            for c in range(x - half_size, x + half_size + 1):
                if 0 <= r < self.HEIGHT and 0 <= c < self.WIDTH:
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

    def update_grid_colors(self, color=(255, 255, 255)):
        # Reset the colors for each grid cell
        for r in range(self.HEIGHT):
            for c in range(self.WIDTH):
                self.grid_colors[r][c] = color
        self._update_canvas_colors()

    def _update_canvas_colors(self):
        # Apply the colors to each rectangle (grid cell) in the canvas
        for r in range(self.HEIGHT):
            for c in range(self.WIDTH):
                grid_color = self.grid_colors[r][c]
                rect_id = (r * self.WIDTH) + c + 1
                self.canvas.itemconfig(rect_id, fill=self.rgb_to_hex(grid_color))

    def rgb_to_hex(self, rgb):
        # Convert RGB tuple to hex string for Tkinter color filling
        return '#%02x%02x%02x' % rgb


    def load_images(self):
        # Using vector graphics to represent agents, obstacles, and target
        rectangle = 'blue_circle'  # Placeholder to represent blue circle as agent
        triangle = 'red_rectangle'  # Placeholder to represent red rectangle as obstacle
        circle = 'green_circle'  # Placeholder to represent green circle as target
        return rectangle, triangle, circle
    
    def get_circle_grid_position(self):
        circle_coords = self.canvas.coords(self.circle)
        grid_position = self.coords_to_state(circle_coords)
        return grid_position

    def move_obstacles(self):
        new_positions = set()  # Set to track new positions of obstacles

        for i, obstacle in enumerate(self.obstacles):
            direction = self.obstacle_directions[i % len(self.obstacle_directions)]
            x_move, y_move = direction[0] * self.UNIT, direction[1] * self.UNIT

            current_coords = self.canvas.coords(obstacle)
            new_x = current_coords[0] + x_move
            new_y = current_coords[1] + y_move

            # Check for grid boundaries
            if new_x < self.UNIT / 2 or new_x > (self.WIDTH - 0.5) * self.UNIT or new_y < self.UNIT / 2 or new_y > (self.HEIGHT - 0.5) * self.UNIT:
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

            # Move the obstacle if the new position is not occupied
            if (new_x, new_y) not in new_positions:
                self.canvas.move(obstacle, x_move, y_move)
                new_positions.add((new_x, new_y))



    def render(self):
        time.sleep(0.000001)
        self.update()

    def destroy_environment(self):
        self.destroy()

    # Method to handle closing event
    def on_closing(self):
        self.destroy_environment()

if __name__ == "__main__":
    env = Env()
    env.mainloop()  # Running the main loop for the Tkinter GUI