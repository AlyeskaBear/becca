"""
Two-dimensional grid task

This task is a 2D extension of the 1D grid world and 
is similar to it in many ways. It is a little more
challenging, because can take two actions to reach a reward state.
"""
import numpy as np
from base_world import World as BaseWorld

class World(BaseWorld):
    """ 
    Two-dimensional grid world

    In this world, the agent steps North, South, East, or West in 
    a 5 x 5 grid-world. Position (4,4) is rewarded and (2,2) 
    is punished. There is also a lesser penalty for each 
    horizontal or vertical step taken. 
    Optimal performance is a reward of about .9 per time step.
    """
    def __init__(self, lifespan=None, test=False):
        """
        Set up the world
        """
        BaseWorld.__init__(self, lifespan)
        self.REWARD_MAGNITUDE = 1.
        self.ENERGY_COST = 0.05 * self.REWARD_MAGNITUDE
        self.JUMP_FRACTION = 0.1
        self.name = 'grid_2D'
        self.name_long = 'two dimensional grid world'
        print "Entering", self.name_long
        self.num_actions = 8
        self.world_size = 5
        self.num_sensors = self.world_size ** 2
        self.world_state = np.array([1, 1])
        # Reward positions (2,2) and (4,4)
        self.targets = [(1,1), (3,3)]
        # Punish positions (2,4) and (4,2)
        self.obstacles = [(1,3), (3,1)]
        self.world_visualize_period = 1e6
        self.brain_visualize_period = 1e3
    
    def step(self, action): 
        """
        Advance the world by one time step
        """
        #print '1', action
        #self.action = np.round(action).ravel().astype(int)
        self.action = action.ravel()
        self.action[np.nonzero(self.action)] = 1.
        self.timestep += 1
        self.world_state += (self.action[0:2] - 
                             self.action[4:6] + 
                             2 * self.action[2:4] -
                             2 * self.action[6:8]).T
        energy = (np.sum(self.action[0:2]) + 
                  np.sum(self.action[4:6]) + 
                  np.sum(2 * self.action[2:4]) +
                  np.sum(2 * self.action[6:8]))
        # At random intervals, jump to a random position in the world
        if np.random.random_sample() < self.JUMP_FRACTION:
            #self.world_state = np.random.random_integers(
            #        0, self.world_size, self.world_state.shape)
            self.world_state = np.random.randint(
                    0, self.world_size, size=self.world_state.size)
        # Enforce lower and upper limits on the grid world 
        # by looping them around
        self.world_state = np.remainder(self.world_state, self.world_size)
        sensors = self.assign_sensors()
        reward = 0
        for obstacle in self.obstacles:
            if tuple(self.world_state) == obstacle:
                reward = - self.REWARD_MAGNITUDE
        for target in self.targets:
            if tuple(self.world_state) == target:
                reward = self.REWARD_MAGNITUDE
        reward -= self.ENERGY_COST * energy
        return sensors, reward

    def assign_sensors(self):
        """ 
        Construct the sensor array from the state information 
        """
        sensors = np.zeros(self.num_sensors)
        sensors[self.world_state[0] + 
                self.world_state[1] * self.world_size] = 1
        return sensors

    def visualize_world(self):
        """ 
        Show the state of the world and the agent 
        """
        print ''.join(['state', str(self.world_state), '  action', 
                       str((self.action[0:2] + 2 * self.action[2:4] - 
                            self.action[4:6] - 2 * self.action[6:8]).T)])
