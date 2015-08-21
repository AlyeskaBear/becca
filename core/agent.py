""" 
the Agent class 
"""
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
mod_path = os.path.dirname(os.path.abspath(__file__))
import arborkey
import drivetrain
import hub
import mainspring
import spindle
import tools

class Agent(object):
    """ 
    A general reinforcement learning agent
    
    It takes in an array of sensor values and 
    a reward and puts out an array of action commands at each time step.
    """
    def __init__(self, num_sensors, num_actions, show=True, 
                 exploit=False, classifier=False, agent_name='test_agent'):
        """
        Configure the Agent

        num_sensors and num_actions are the only absolutely necessary
        arguments. They define the number of elements in the 
        sensors and actions arrays that the agent and the world use to
        communicate with each other. 

        Keyword arguments:
        _________________
        classifier : bool
            If True, this arg indicates that the agent is participating 
            in a classification task. Certain parameters are selected
            specifically to maximize performance, since classification tasks
            are a special subset of all reinforcement learning tasks.
        exploit : bool
            If True, this arg indicates that the agent should try to maximize
            performance in the near term, rather than invest effort in learning.
            This is useful when benchmarking performance on standard 
            learning tasks.
        """
        self.num_sensors = num_sensors
        # Add one to create a null action. 
        # This action has no effect on the world.
        self.num_actions = num_actions + 1
        # Force display of progress and block the agent?
        self.show = show
        # Number of time steps between generating visualization plots
        self.display_interval = 1e4
        # Number of time steps between making a backup copy of the agent
        self.backup_interval = self.display_interval
        self.name = agent_name
        self.log_dir = os.path.normpath(os.path.join(mod_path, '..', 'log'))
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.pickle_filename = os.path.join(
                self.log_dir, '.'.join([agent_name, 'pickle']))

        # Initialize agent infrastructure.
        min_cables = 5 * self.num_sensors

        self.drivetrain = drivetrain.Drivetrain(min_cables, exploit=exploit,
                                                classifier=classifier)
        num_cables = self.drivetrain.cables_per_ziptie
        self.hub = hub.Hub(num_cables, num_actions=self.num_actions, 
                           num_sensors=self.num_sensors,
                           exploit=exploit, classifier=classifier,
                           name='_'.join([self.name, 'hub']))
        self.spindle = spindle.Spindle(num_cables)
        self.mainspring = mainspring.Mainspring(num_cables, self.num_actions,
                name='_'.join([self.name, 'mainspring']))
        self.arborkey = arborkey.Arborkey(num_cables)
        self.arborkey_goal = None
        self.attended_index = None
        self.action = np.zeros((self.num_actions,1))
        self.cumulative_reward = 0
        self.time_since_reward_log = 0 
        self.reward_history = []
        self.reward_steps = []
        self.reward_max = 1.
        self.timestep = 0
        self.graphing = True
        self.frames_directory = os.path.join('core', 'hierarchy_frames')
        self.frame_counter = 10000

    def step(self, sensors, reward):
        """
        Advance by one time step. Take sensor and reward data in, 
        learn from them, and use them to choose an action.
        """
        """
        Scale the reward by its greatest-ever observed magnitude 
        so that it falls between -1 and 1.
        A reward of zero is always zero.
        """  
        if np.abs(reward) > self.reward_max:
            self.reward_max = np.abs(reward)
        self.reward = reward / self.reward_max

        self.timestep += 1
        if sensors.ndim == 1:
            sensors = sensors[:,np.newaxis]
        # Propogate the new sensor inputs through the drivetrain
        #feature_activities = self.drivetrain.step_up(self.action, sensors)
        # debug: send sensors only, not actions
        #import time
        #tic = time.clock()
        (feature_activities, modified_cables) = self.drivetrain.step_up(
                self.action, sensors)
        #toc = time.clock()
        #print toc - tic

        # The drivetrain will grow over time as the agent gains experience.
        # If the drivetrain added a new ziptie, scale the hub up appropriately.
        if self.drivetrain.ziptie_added:
            self.hub.add_cables(self.drivetrain.cables_per_ziptie)
            self.spindle.add_cables(self.drivetrain.cables_per_ziptie)
            self.mainspring.add_cables(self.drivetrain.cables_per_ziptie)
            self.drivetrain.ziptie_added = False
        # Feed the feature_activities to the hub for calculating goals
        self.hub_goal, hub_reward, hub_curiosity, reward_trace = self.hub.step(
                feature_activities, self.reward, modified_cables) 
        # debug: toggle attention on and off
        use_attention = False 
        if use_attention:
            # Evaluate the goal using the mainspring
            mainspring_reward = self.mainspring.evaluate(self.hub_goal) 
            # Choose a single feature to attend 
            (self.attended_index, attended_activity) = self.spindle.step(
                    feature_activities)
            # Incorporate the intended feature into short- and long-term memory
            self.mainspring.step(self.attended_index, 
                                 attended_activity, 
                                 reward_trace)
            # Pass the hub goal on to the arborkey for further evaluation
            self.arborkey_goal = self.arborkey.step(self.hub_goal, 
                                                    hub_reward,
                                                    hub_curiosity,
                                                    mainspring_reward, 
                                                    self.reward)
            self.hub.update(feature_activities, self.arborkey_goal, self.reward)
            self.mainspring.update(self.arborkey_goal)
            #if self.arborkey_goal is not None:
            #    self.drivetrain.assign_goal(self.arborkey_goal)
            action_index = self.arborkey_goal
        else:
            # debug: circumvent attention (spindle, mainspring, arborkey)
            self.hub.update(feature_activities, self.hub_goal, self.reward)
            #if self.hub_goal is not None:
            #    self.drivetrain.assign_goal(self.hub_goal)
            action_index = self.hub_goal

        # Choose an action
        #self.action = self.drivetrain.step_down()
        self.action = np.zeros(self.action.shape)
        if action_index is not None:
            # debug: Allow fractional actions?
            #self.action[action_index] = hub_reward + hub_curiosity
            self.action[action_index] = 1.
        # debug: Choose a single random action 
        random_action = False
        if random_action:
            self.action = np.zeros(self.action.shape)
            random_action_index = np.random.randint(self.action.size)
            self.action[random_action_index] = 1. 

        if (self.timestep % self.backup_interval) == 0:
                self._save()    
        # Log the reward
        self.cumulative_reward += self.reward
        self.time_since_reward_log += 1
        if np.mod(self.timestep, self.display_interval) == 0.:
            self.visualize()
        return self.action

    def get_index_projections(self, to_screen=False):
        """
        For each feature the agent has extracted, find a representation
        in terms of low level sensors and actions.
        """
        return self.drivetrain.get_index_projections(to_screen=to_screen)

    def visualize(self):
        """ 
        Show the current state and some history of the agent 
        """
        print ' '.join([self.name, 'is', str(self.timestep), 'time steps old'])
        self.reward_history.append(float(self.cumulative_reward) / 
                                   (self.time_since_reward_log + 1))
        self.cumulative_reward = 0    
        self.time_since_reward_log = 0
        self.reward_steps.append(self.timestep)
        self._show_reward_history()

        # TODO: Some of these still need to be created
        tools.visualize_hierarchy(self, show=False)
        self.drivetrain.visualize()
        #self.spindle.visualize()
        tools.visualize_hub(self.hub, show=False)
        #tools.visualize_mainspring(self.mainspring, show=False)
        #self.mainspring.visualize()
        #self.arborkey.visualize()
 
    def report_performance(self):
        """
        Make a report of how the agent did over its lifetime
        """
        performance = np.mean(self.reward_history)
        print("Final performance is %f" % performance)
        self._show_reward_history(hold_plot=self.show)
        return performance
    
    def _show_reward_history(self, hold_plot=False, filename=None):
        """ 
        Show the agent's reward history and save it to a file 
        """
        if self.graphing:
            fig = plt.figure(1)
            plt.plot(self.reward_steps, self.reward_history)
            plt.xlabel("time step")
            plt.ylabel("average reward")
            plt.title(''.join(('Reward history for ', self.name)))
            fig.show()
            fig.canvas.draw()
            if filename is None:
                filename = ''.join(['reward_history_', self.name ,'.png'])
                pathname = os.path.join(self.log_dir, filename)
            plt.savefig(pathname, format='png')
            if hold_plot:
                plt.show()
    
    def _save(self):
        """ 
        Archive a copy of the agent object for future use 
        """
        success = False
        make_backup = True
        print "Attempting to save agent..."
        try:
            with open(self.pickle_filename, 'wb') as agent_data:
                pickle.dump(self, agent_data)
            if make_backup:
                with open(''.join((self.pickle_filename, '.bak')), 
                          'wb') as agent_data_bak:
                    pickle.dump(self, agent_data_bak)
            print("Agent data saved at " + str(self.timestep) + " time steps")
        except IOError as err:
            print("File error: " + str(err) + 
                  " encountered while saving agent data")
        except pickle.PickleError as perr: 
            print("Pickling error: " + str(perr) + 
                  " encountered while saving agent data")        
        else:
            success = True
        return success
        
    def restore(self):
        """ 
        Reconstitute the agent from a previously saved agent 
        """
        restored_agent = self
        try:
            with open(self.pickle_filename, 'rb') as agent_data:
                loaded_agent = pickle.load(agent_data)
            """
            Compare the number of channels in the restored agent with 
            those in the already initialized agent. If it matches, 
            accept the agent. If it doesn't,
            print a message, and keep the just-initialized agent.
            """
            if ((loaded_agent.num_sensors == self.num_sensors) and 
               (loaded_agent.num_actions == self.num_actions)):
                print(''.join(('Agent restored at timestep ', 
                               str(loaded_agent.timestep),
                               ' from ', self.pickle_filename)))
                restored_agent = loaded_agent
            else:
                print("The agent " + self.pickle_filename + " does not have " +
                      "the same number of input and output elements as " + 
                      "the world.")
                print("Creating a new agent from scratch.")
        except IOError:
            print("Couldn't open %s for loading" % self.pickle_filename)
        except pickle.PickleError, e:
            print("Error unpickling world: %s" % e)
        return restored_agent
