# AdvAI_Project
Title: **Single-Agent to Swarm: Proximal Policy Optimization (PPO)-Based Reinforcement Learning for Autonomous Tile Cleaning**

**Our project requires Webots software and python with the libraries used in the code installed.**

**Single Agent Code (Single_Agent Folder)**
- The subfolder named "Webots_1_Agent" is the code used for simulating the reinforcement learning using PPO for a single e-puck robot on webots.
- The subfolder named "Webots_1_Agent_Episode_vs_Timestep" is for plotting the Episodes vs Cumulative Timesteps with a maximum of 50k time steps if the episode does not end with the robot cleaning all the tiles.
- A notebook that creates some of the plots used in this project. Also note that some of the plots used in this project were created from the code in webots itself.
- In general the supervisor code is used to manage the policy and the environment creation, while the robot controller is for moving the robot.

**Multiagent Code (Swarm Folder)**
Both files contain the codes and environments to accomodate multiple bots to make a swarm.
The main difference is that AIMo has mostly the same training code, with the addition of sensors to allow for collision avoidance (punishment for exceeding a certain value).
The other folder contains code that does not have these sensors, but is meant to have each episode have a maximum of 50k steps, with learning occuring every 1k.

**More information about the details of our implemntation can be found in the paper uploaded.**
**A youtube video that contains the demo and a brief presentation can be found on this link:** 
