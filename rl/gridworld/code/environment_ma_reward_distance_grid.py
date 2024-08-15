import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from heapq import heappop, heappush

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

        self.render()
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

            observations.append(observation)

        return observations

    def step(self, actions):
        rewards = []
        dones = []
        next_states = []

        circle_pos = self.get_circle_grid_position()

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
            state = agent['coords']
            base_action = np.array([0, 0])
            message = None

            if self.is_agent_silent:
                if action[0] == 1:  # up
                    if state[1] > UNIT:
                        base_action[1] -= UNIT
                elif action[0] == 2:  # down
                    if state[1] < (HEIGHT - 1) * UNIT:
                        base_action[1] += UNIT
                elif action[0] == 3:  # left:
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

            # Calculate the shortest path distance before moving
            initial_pos = self.coords_to_state(state)
            initial_distance = self.astar(initial_pos, circle_pos)

            # Move the agent and update its state
            self.canvas.move(agent['image_obj'], base_action[0], base_action[1])
            self.canvas.tag_raise(agent['image_obj'])
            next_state = self.canvas.coords(agent['image_obj'])
            agent['coords'] = next_state

            # Calculate the shortest path distance after moving
            new_pos = self.coords_to_state(next_state)
            new_distance = self.astar(new_pos, circle_pos)

            # Determine reward based on distance reduction
            reward = initial_distance - new_distance

            # Check if agent reaches the target
            done = False
            if new_pos == circle_pos:
                reward = 100
                done = True

            # Append reward and done status
            rewards.append(reward)
            dones.append(done)

            # Prepare next state observation
            next_state_obs = self.coords_to_state(next_state)

            # Append received message to observation if communication is enabled
            if not self.is_agent_silent:
                next_state_comms = []
                for other_agent in self.agents:
                    if other_agent == agent:
                        continue  # Skip the current agent itself

                    other_agent_message = actions[other_agent['id']][1]
                    next_state_comms.append(other_agent_message)

            next_state_observation = [next_state_obs, done, next_state_comms if not self.is_agent_silent else None]

            next_states.append(next_state_observation)

        # Check if all agents are done
        if all(dones):
            self.update_grid_colors((0, 255, 0))

        return next_states, rewards, dones

    def render(self):
        time.sleep(0.5)
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

    def astar(self, start, goal):
        """A* algorithm to find the shortest path from start to goal."""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                return g_score[current]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g_score = g_score[current] + 1

                if 0 <= neighbor[0] < WIDTH and 0 <= neighbor[1] < HEIGHT:
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heappush(open_set, (f_score[neighbor], neighbor))

        return float('inf')

    @staticmethod
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb


if __name__ == "__main__":
    env = Env()
    env.mainloop()
