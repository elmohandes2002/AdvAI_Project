from controller import Supervisor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import csv
import matplotlib.pyplot as plt

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

class CleaningSupervisor(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        # Environment
        self.cell_size = 0.2
        self.grid_size = 10
        self.world_size = self.cell_size * self.grid_size

        # Action space
        self.actions = [
            (0.5, 0.0),    # Forward
            (0.0, 0.5),    # Turn left
            (0.0, -0.5),   # Turn right
        ]

        # PPO setup
        self.state_dim = 2
        self.action_dim = len(self.actions)
        self.policy = Policy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.entropy_coeff = 0.01
        self.max_episodes = 300

        # Exploration schedule
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

        # Tracking
        self.episode = 0
        self.reward_history = []
        self.cleaned_history = []

        # Robot setup
        self.robot_node = self.getFromDef("CLEANING_ROBOT")
        self.emitter = self.getDevice('emitter')
        self.receiver = self.getDevice('receiver')
        self.receiver.enable(self.timestep)

        # Grid setup
        self.children_field = self.getRoot().getField('children')
        self.original_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.initialize_environment()
        self.reset_episode()

    def initialize_environment(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x + y) % 3 == 0 and not (4 <= x <= 5 and 4 <= y <= 5):
                    self.original_grid[x, y] = 1  # Dirt
                if (x == 2 and y == 2) or (x == 7 and y == 7):
                    self.original_grid[x, y] = 2  # Obstacle

    def _create_tile(self, x, y, dirt=True):
        pos_x = (x * self.cell_size) - (self.world_size / 2) + (self.cell_size / 2)
        pos_y = (y * self.cell_size) - (self.world_size / 2) + (self.cell_size / 2)
        height = 0.01 if dirt else 0.04
        color = "0 0 0" if dirt else "0.8 0.1 0.1"

        if dirt:
            proto = f"""
            DEF TILE_{x}_{y} Transform {{
              translation {pos_x} {pos_y} {height/2}
              children [
                Shape {{
                  appearance Appearance {{
                    material Material {{ diffuseColor {color} }}
                  }}
                  geometry Box {{ size {self.cell_size} {self.cell_size} {height} }}
                }}
              ]
            }}
            """
        else:
            proto = f"""
            DEF TILE_{x}_{y} Solid {{
              translation {pos_x} {pos_y} {height/2}
              children [
                Shape {{
                  appearance Appearance {{
                    material Material {{ diffuseColor {color} }}
                  }}
                  geometry Box {{ size {self.cell_size} {self.cell_size} {height} }}
                }}
              ]
              boundingObject Box {{ size {self.cell_size} {self.cell_size} {height} }}
              physics Physics {{ }}
            }}
            """
        self.children_field.importMFNodeFromString(-1, proto)

    def reset_environment(self):
        # Remove old tiles
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                node = self.getFromDef(f"TILE_{x}_{y}")
                if node:
                    node.remove()

        # Recreate based on original grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.original_grid[x, y] == 1:
                    self._create_tile(x, y, dirt=True)
                elif self.original_grid[x, y] == 2:
                    self._create_tile(x, y, dirt=False)

        self.grid = self.original_grid.copy()

    def reset_episode(self):
        safe_positions = []
        for x in range(1, self.grid_size-1):
            for y in range(1, self.grid_size-1):
                if self.original_grid[x, y] == 0:
                    safe_positions.append((x, y))

        start_x, start_y = random.choice(safe_positions)
        pos_x = (start_x * self.cell_size) - (self.world_size / 2) + (self.cell_size / 2)
        pos_y = (start_y * self.cell_size) - (self.world_size / 2) + (self.cell_size / 2)

        self.robot_node.getField("translation").setSFVec3f([pos_x, pos_y, 0.05])
        self.robot_node.getField("rotation").setSFRotation([0, 0, 1, random.uniform(0, 2*np.pi)])

        self.reset_environment()

        self.cleaned_dirt = 0
        self.step_counter = 0
        self.episode_reward = 0
        self.visited_tiles = set()
        self.last_positions = deque(maxlen=5)
        self.memory = []

    def get_robot_grid_pos(self):
        pos = self.robot_node.getPosition()
        grid_x = int((pos[0] + self.world_size/2) / self.cell_size)
        grid_y = int((pos[1] + self.world_size/2) / self.cell_size)
        return (grid_x, grid_y)

    def get_state(self):
        x, y = self.get_robot_grid_pos()
        return np.array([x/self.grid_size, y/self.grid_size], dtype=np.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item()

    def reward_function(self, next_pos):
        x, y = next_pos
        reward = -0.1

        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return -5.0

        if self.grid[int(x), int(y)] == 2:
            return -2.0

        if self.grid[int(x), int(y)] == 1:
            self.grid[int(x), int(y)] = 0
            self.cleaned_dirt += 1
            reward += 5.0
            self._clear_tile(int(x), int(y))

        if (x, y) not in self.visited_tiles:
            reward += 0.2
            self.visited_tiles.add((x, y))

        self.last_positions.append((x, y))
        if len(self.last_positions) == 5 and len(set(self.last_positions)) == 1:
            reward -= 0.5

        return reward

    def _clear_tile(self, x, y):
        node = self.getFromDef(f"TILE_{x}_{y}")
        if node:
            node.remove()

    def update_policy(self):
        if len(self.memory) < 64:
            return

        states = torch.tensor(np.array([t[0] for t in self.memory]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in self.memory]), dtype=torch.long)
        old_log_probs = torch.tensor(np.array([t[2] for t in self.memory]), dtype=torch.float32)
        rewards = [t[3] for t in self.memory]

        discounted_rewards = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        for _ in range(4):
            probs, state_values = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions)

            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            advantages = discounted_rewards - state_values.squeeze().detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() + 0.5 * advantages.pow(2).mean() - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()

    def save_results(self):
        with open('results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Total Reward', 'Tiles Cleaned'])
            for i in range(len(self.reward_history)):
                writer.writerow([i, self.reward_history[i], self.cleaned_history[i]])

    def plot_results(self):
        plt.plot(self.reward_history)
        plt.title('Reward vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid()
        plt.show()

    def run(self):
        print("Starting training...")
        while self.step(self.timestep) != -1 and self.episode < self.max_episodes:
            state = self.get_state()
            robot_grid = self.get_robot_grid_pos()

            action_idx = self.choose_action(state)
            action = self.actions[action_idx]

            self.emitter.send(f"{action[0]},{action[1]}".encode())

            start_pos = robot_grid
            moved = False
            wait_steps = 0
            while self.step(self.timestep) != -1:
                current_pos = self.get_robot_grid_pos()
                if current_pos != start_pos:
                    moved = True
                    break
                wait_steps += 1
                if wait_steps > 100:
                    break

            next_state = self.get_state()
            next_grid = self.get_robot_grid_pos()
            reward = self.reward_function(next_grid)

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs, _ = self.policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor(action_idx))
            self.memory.append((state, action_idx, log_prob.item(), reward))

            self.episode_reward += reward
            self.step_counter += 1

            if self.cleaned_dirt >= np.sum(self.original_grid == 1) or self.step_counter >= 1000:
                self.update_policy()
                self.reward_history.append(self.episode_reward)
                self.cleaned_history.append(self.cleaned_dirt)
                print(f"Episode {self.episode} finished. Total reward: {self.episode_reward:.2f}, Cleaned: {self.cleaned_dirt} tiles")

                # Epsilon decay
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

                self.episode += 1
                self.reset_episode()

        self.save_results()
        self.plot_results()

if __name__ == "__main__":
    supervisor = CleaningSupervisor()
    supervisor.run()
