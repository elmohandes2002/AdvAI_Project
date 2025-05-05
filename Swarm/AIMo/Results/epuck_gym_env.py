import gymnasium as gym
import numpy as np
import random
import struct
import time
import json
from cleaning_supervisor import CleaningSupervisor


class EpuckCleaningEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 10}
    
    def __init__(self, supervisor: CleaningSupervisor = None):
        self.visualize = False  # Set this False during actual training
        super(EpuckCleaningEnv, self).__init__()
        # if Webots has already spawned a CleaningSupervisor, reuse it…
        if supervisor is not None:
            self.supervisor = supervisor
        else:
            # …otherwise create your own (this only happens if you import/run epuck_gym_env
            # from a standalone Python script outside Webots, which you won’t do here)
            self.supervisor = CleaningSupervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
    
        # --- Action Space (5 actions)
        self.actions = ['north', 'south', 'east', 'west', 'clean']
        self.action_space = gym.spaces.Discrete(len(self.actions))

        # --- Observation Space (10D: 8 distance sensors + 2 GPS)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # --- Communication
        self.receiver = self.supervisor.getDevice("receiver")
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver.setChannel(1)
        self.emitter.setChannel(1)
        self.receiver.enable(self.timestep)

        # --- Internal state
        self.dirt_threshold =650 #a starting avg value for the ground sensors to detect dirt
        self.last_position = None
        self.CLEANED=0
        self.TOTALREW=0
        self.steps = 0
        self.max_steps = 1000  # Max steps per episode
        self.steps_per_action = 16 #having more steps per action bcz each step in sim is ~32ms 
        #which is barely anything for the bot (and us to view)

    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)
        self.supervisor.reset_environment()
        
        if self.visualize:
            # keep it at real-time (about 1× speed)
            self.supervisor.simulationSetMode(
                self.supervisor.SIMULATION_MODE_REAL_TIME
            )
        else:
            # run as fast as possible without rendering
            self.supervisor.simulationSetMode(2)

        # Clear old packets
        while self.receiver.getQueueLength() > 0:
            self.receiver.nextPacket()
        
        print(f"Episode done | Cleaned {self.CLEANED}/{self.supervisor.DIRTY_SPACES} | Steps: {self.steps} | Obstacles {self.supervisor.OBSTACLES}")
        self.supervisor._save_evaluation(self.steps,self.CLEANED, self.TOTALREW)
        
        self.CLEANED=0
        self.last_position = None
        self.steps = 0
        self.TOTALREW=0
        

        
        # Do one step to receive first position
        self.supervisor.step(self.timestep)
        self._receive_robot_position()

        obs = self._get_observation()
        return obs, {}

    def step(self, action_idx):
        action = self.actions[action_idx]
        self.emitter.send(action.encode('utf-8'))

        reward = 0.0
        done = False
        self.cleaned_this_action = False #using this to limit rewards
        #to once per action rather than every timestep
        for _ in range(self.steps_per_action):
            self.supervisor.step(self.timestep)
            self._receive_robot_position()
            reward += self._compute_reward(action)
            if self._check_all_clean():
                done = True
                print("CLEARED ALL!!")
                break

        obs = self._get_observation()
        self.steps += 1
        
        self.TOTALREW+=reward
        
        if self.steps >= self.max_steps:
            done = True
        # consider truncation or max_steps here if you want
        return obs, reward, done, False, {}

    def render(self):
        # Nothing needed, Webots handles visualization
        #gym needs this class though so we keepin it
        pass

    def close(self):
        # self.supervisor.simulationQuit(0)
        pass

    # --- Helper Functions ---

    def _receive_robot_position(self):
        """Receive latest robot position."""
        while self.receiver.getQueueLength() > 0:
            data = self.receiver.getString()
            packet = json.loads(data)
            x, y = packet["pos"]
            self.last_position = (x, y)
            self.last_sensor_values = np.array(packet["sensors"], dtype=np.float32)
            self.last_ground_values = np.array(packet["ground"], dtype=np.float32)
            
            self.receiver.nextPacket()
            
    def _get_observation(self):
        """Generate real observation vector from robot sensors."""
        if hasattr(self, 'last_sensor_values'):
                sensors = np.clip(self.last_sensor_values / 1000.0, 0, 1)
        else:
                sensors = np.zeros(8)
    
        # Get GPS values
        gps_x, gps_y = 0.5, 0.5  # default center if not found
    
        if self.last_position:
            gps_x = (self.last_position[0] + (self.supervisor.world_size/2)) / self.supervisor.world_size
            gps_y = (self.last_position[1] + (self.supervisor.world_size/2)) / self.supervisor.world_size
    
        gps_pos = np.array([gps_x, gps_y])
    
        # Final observation
        obs = np.concatenate([sensors, gps_pos]).astype(np.float32)
        return obs

    def _compute_reward(self, action=None):
        """Compute reward based on action, sensors, and ground detection."""
        

        reward = -0.01  # Small penalty for each timestep (encourage efficiency)
    #self.grid[robot_x, robot_y] == self.DIRT:
        # Check for cleaning actions
        if action is not None and action == "clean" and not self.cleaned_this_action:
            #check environment's tile
            robot_node = self.supervisor.getFromDef("CLEANING_ROBOT")
            if robot_node:
                pos = robot_node.getPosition()
                grid_x = int((pos[0] + (self.supervisor.world_size / 2)) / self.supervisor.cell_size)
                grid_y = int((pos[1] + (self.supervisor.world_size / 2)) / self.supervisor.cell_size)
    
                # Clip to safe grid bounds
                grid_x = np.clip(grid_x, 0, self.supervisor.grid_width - 1)
                grid_y = np.clip(grid_y, 0, self.supervisor.grid_height - 1)
    
                tile_is_dirty = (self.supervisor.grid[grid_x, grid_y] == self.supervisor.DIRT)
            else:
                tile_is_dirty = False
                
            if hasattr(self, 'last_ground_values'):
                if np.mean(self.last_ground_values) < self.dirt_threshold and tile_is_dirty: #detected, dirty =good
                    print("[Reward] Successful cleaning!")
                    reward += 1.0  # Positive reward for cleaning dirt
                    self.CLEANED+=1
                    print(f"{np.mean(self.last_ground_values)} and tile {tile_is_dirty} threshold {self.dirt_threshold}")
                elif np.mean(self.last_ground_values)<self.dirt_threshold and not tile_is_dirty: #detected, not dirty =bad
                    self.dirt_threshold *= 0.999 #decreasing threshold so false positive less likely
                    print(f"[Reward] False Positive detected! {np.mean(self.last_ground_values)} and tile {tile_is_dirty} threshold {self.dirt_threshold}")

                elif np.mean(self.last_ground_values)>=self.dirt_threshold and tile_is_dirty:
                    self.dirt_threshold *= 1.001 #increasing threshold so false negative less likely 
                    print(f"[Reward] False Negative detected! {np.mean(self.last_ground_values)} and tile {tile_is_dirty} threshold {self.dirt_threshold}")
                else:
                    print(f"[Reward] Tried to clean clean floor!")
                    reward -= 1.0  # Penalty for cleaning clean tile (detected as clean AND is clean)
            self.supervisor._clean_cell(grid_x, grid_y)
            self.cleaned_this_action=True

        # Obstacle penalties
        if hasattr(self, 'last_sensor_values'):
            sensors = np.clip(self.last_sensor_values / 1000.0, 0, 1)
            # print(f"sensors {sensors}")
    
                # Far obstacle detected
            if np.any((sensors > 0.3) & (sensors <= 0.6)):
                reward -= 0.2
    
            # Very close obstacle detected
            if np.any((sensors > 0.6) & (sensors <= 0.9)):
                reward -= 0.5
            #might as well be crashing  
            if np.any(sensors > 0.9):
               reward -= 0.8
    
        return reward

    def _check_all_clean(self):
        """Check if all dirt is cleaned."""
        return np.sum(self.supervisor.grid == self.supervisor.DIRT) == 0

    def _world_to_grid(self, pos):
        """Convert world coordinates to grid indices."""
        x, y = pos
        grid_x = int((x + (self.supervisor.world_size / 2)) / self.supervisor.cell_size)
        grid_y = int((y + (self.supervisor.world_size / 2)) / self.supervisor.cell_size)
        return grid_x, grid_y
