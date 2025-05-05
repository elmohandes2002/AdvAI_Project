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
        
        self.num_robots= 2
        
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
        self.policies = [Policy(self.state_dim, self.action_dim) for _ in range(self.num_robots)]
        self.optimizers = [optim.Adam(p.parameters(), lr=3e-4) for p in self.policies]
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.entropy_coeff = 0.01
        self.max_episodes = 600
        self.max_steps=1000

        # Exploration schedule
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995
        self.epsilons = [self.epsilon_start] * self.num_robots

        # Tracking
        self.episode = 0
        self.reward_histories = [[] for _ in range(self.num_robots)]
        self.cleaned_histories = [[] for _ in range(self.num_robots)]
        self.memories = [[] for _ in range(self.num_robots)]

        # Robot setup
        self.robot_nodes = [self.getFromDef(f"CLEANING_ROBOT_{i+1}") for i in range(self.num_robots)]
        self.emitter = self.getDevice('emitter')
        self.emitter.setChannel(1) 
        self.receiver = self.getDevice('receiver')
        self.receiver.enable(self.timestep)
        self.robot_sensor_readings = {i: [0.0]*8 for i in range(self.num_robots)}


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
        # Fixed set of 8 obstacle positions (ensure they are valid and not overlapping with center)
        # obstacle_positions = [(2, 2), (17, 17), (5, 14), (14, 5), (3, 10), (10, 3), (7, 17), (17, 7)]
        obstacle_positions=[(2,2),(7,7)]
        for x, y in obstacle_positions:
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
        starts = random.sample(safe_positions, self.num_robots)


        for i in range(self.num_robots):
            x, y = starts[i]
            pos_x = (x * self.cell_size) - (self.world_size / 2) + (self.cell_size / 2)
            pos_y = (y * self.cell_size) - (self.world_size / 2) + (self.cell_size / 2)
            self.robot_nodes[i].getField("translation").setSFVec3f([pos_x, pos_y, 0.05])
            self.robot_nodes[i].getField("rotation").setSFRotation([0, 0, 1, random.uniform(0, 2*np.pi)])

        self.reset_environment()

        self.cleaned_dirt = [0]* self.num_robots
        self.step_counters = [0]* self.num_robots
        self.episode_rewards =[0] * self.num_robots
        self.visited_tiles = [set() for _ in range(self.num_robots)]
        self.last_positions = [deque(maxlen=5) for _ in range(self.num_robots)]
        self.memories = [[] for _ in range(self.num_robots)]
        self.robot_sensor_readings = {i: [0.0]*8 for i in range(self.num_robots)}


    def get_robot_grid_pos(self, idx):
        pos = self.robot_nodes[idx].getPosition()
        grid_x = int((pos[0] + self.world_size/2) / self.cell_size)
        grid_y = int((pos[1] + self.world_size/2) / self.cell_size)
        return (grid_x, grid_y)

    def get_state(self, idx):
        x, y = self.get_robot_grid_pos(idx)
        return np.array([x / self.grid_size, y / self.grid_size], dtype=np.float32)

    def choose_action(self, state, idx):
        if random.random() < self.epsilons[idx]:
            return random.randint(0, self.action_dim - 1)
    
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.policies[idx](state_tensor)
            dist = torch.distributions.Categorical(probs)
            return dist.sample().item()

    def reward_function(self, x, y, idx):
        reward = -0.1

        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size): #shouldn't be possible
            return -5.0

        if self.grid[int(x), int(y)] == 2:
            return -2.0

        if self.grid[int(x), int(y)] == 1:
            self.grid[int(x), int(y)] = 0
            self.cleaned_dirt[idx] += 1
            reward += 5.0
            self._clear_tile(int(x), int(y))

        if (x, y) not in self.visited_tiles[idx]:
            reward += 0.2
            self.visited_tiles[idx].add((x, y))

        self.last_positions[idx].append((x, y))
        if len(self.last_positions[idx]) == 5 and len(set(self.last_positions[idx])) == 1:
            reward -= 0.5
            
        sensor_values = self.robot_sensor_readings.get(idx, [0.0]*8) #proximity
        if any(v > 1500.0 for v in sensor_values):  
            reward -= 0.5
            
        # Collision penalty if another robot is in the same grid cell
        # for other_idx in range(self.num_robots):
            # if other_idx != idx:
                # other_pos = self.get_robot_grid_pos(other_idx)
                # if (int(x), int(y)) == (int(other_pos[0]), int(other_pos[1])):
                    # reward -= 1.0  # basic collision penalty
                    # break

        return reward

    def _clear_tile(self, x, y):
        node = self.getFromDef(f"TILE_{x}_{y}")
        if node:
            node.remove()

    def update_policy(self, idx):
        mem = self.memories[idx]
        if len(mem) < 64:
            return

        states = torch.tensor(np.array([t[0] for t in self.memories[idx]]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in self.memories[idx]]), dtype=torch.long)
        old_log_probs = torch.tensor(np.array([t[2] for t in self.memories[idx]]), dtype=torch.float32)
        rewards = [t[3] for t in self.memories[idx]]

        discounted_rewards = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        for _ in range(4):
            probs, state_values = self.policies[idx](states)
            dist = torch.distributions.Categorical(probs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions)

            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            advantages = discounted_rewards - state_values.squeeze().detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() + 0.5 * advantages.pow(2).mean() - self.entropy_coeff * entropy

            self.optimizers[idx].zero_grad()
            loss.backward()
            self.optimizers[idx].step()

        self.memories[idx].clear()

    def save_results(self):
        with open('results2.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Episode']
            for i in range(self.num_robots):
                header.extend([f'Robot{i+1}_Reward', f'Robot{i+1}_Cleaned'])
            writer.writerow(header)
            for ep in range(len(self.reward_histories[0])):
                row = [ep]
                for i in range(self.num_robots):
                    row.extend([self.reward_histories[i][ep], self.cleaned_histories[i][ep]])
                writer.writerow(row)

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        for i in range(self.num_robots):
            ax1.plot(self.reward_histories[i], label=f'Robot {i+1}')
            ax2.plot(self.cleaned_histories[i], label=f'Robot {i+1}')

        ax1.set_title('Reward vs Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid()
        ax1.legend()

        ax2.set_title('Tiles Cleaned vs Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Tiles Cleaned')
        ax2.grid()
        ax2.legend()

        plt.tight_layout()
        plt.show()
        
    def receive_sensor_data(self):
        while self.receiver.getQueueLength() > 0:
            try:
                message = self.receiver.getString().decode('utf-8')  # e.g., "2:val0,val1,...,val7"
                self.receiver.nextPacket()
    
                if ':' not in message:
                    continue  # skip malformed message
    
                robot_id_str, sensor_str = message.split(':', 1)
                robot_id = int(robot_id_str)
    
                sensor_values = list(map(float, sensor_str.strip().split(',')))
    
                if len(sensor_values) != 8:
                    continue  # expected exactly 8 proximity values
    
                self.robot_sensor_readings[robot_id] = sensor_values
    
            except (ValueError, IndexError):
                continue  # skip any bad messages silently
        
    def run(self):
        print("Starting training...")
        while self.step(self.timestep) != -1 and self.episode < self.max_episodes:
            for i in range(self.num_robots):
                state = self.get_state(i)
                action_idx = self.choose_action(state, i)
                action = self.actions[action_idx]
                self.emitter.send(f"{i}:{action[0]},{action[1]}".encode())

                start_pos = self.get_robot_grid_pos(i)
                wait_steps = 0
                while self.step(self.timestep) != -1:
                    if self.get_robot_grid_pos(i) != start_pos:
                        break
                    wait_steps += 1
                    if wait_steps > 100:
                        break

                x, y = self.get_robot_grid_pos(i)
                reward = self.reward_function(x, y, i)

                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                probs, _ = self.policies[i](state_tensor)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(torch.tensor(action_idx))
                self.memories[i].append((state, action_idx, log_prob.item(), reward))

                self.episode_rewards[i] += reward
                self.step_counters[i] += 1
                
            total_dirt = np.sum(self.original_grid == 1)
            if sum(self.cleaned_dirt) >= total_dirt or all(s >= self.max_steps for s in self.step_counters):
                for i in range(self.num_robots):
                    self.update_policy(i)
                    self.reward_histories[i].append(self.episode_rewards[i])
                    self.cleaned_histories[i].append(self.cleaned_dirt[i])
                    self.epsilons[i] = max(self.epsilons[i] * self.epsilon_decay, self.epsilon_end)

                print(f"Episode {self.episode} done. Rewards: {self.episode_rewards}, Cleaned: {self.cleaned_dirt}")
                self.episode += 1
                self.reset_episode()

        self.save_results()
        self.plot_results()


if __name__ == "__main__":
    supervisor = CleaningSupervisor()
    supervisor.run()
