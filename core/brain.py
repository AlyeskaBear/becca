""" 
The Brain class. 
"""

import cPickle as pickle
#import numpy as np
import os
# Identify the full local path of the brain.py module.  
# This trick is used to conveniently locate other BECCA resources.
mod_path = os.path.dirname(os.path.abspath(__file__))

import amygdala
import cerebellum
import ganglia
import cortex
import tools

class Brain(object):
    """ 
    A biologically motivated learning algorithm.
    
    The components are described as brain regions, but beware that any such
    description is highly controversial among functional and computational
    neuroscientists. This model is a collection of my own best guesses
    at this point in time and doesn't represent any kind 
    of consensus or orthodoxy in the field.

    Attributes
    ----------
    amygdala, cerebellum, cortex, ganglia: 
        Refer the documentation in each of these respective modules.
    backup_interval : int
        The number of time steps between saving a copy of the ``brain``
        out to a pickle file for easy recovery.
    log_dir : str
        Relative path to the ``log`` directory. This is where backups
        and images of the ``brain``'s state and performance are kept.
    name : str
        Unique name for this ``brain``.
    num_actions : int
        The number of distinct actions that the ``brain`` can choose to 
        execute in the world.
    num_sensors : int
        The number of distinct sensors that the world will be passing in 
        to the ``brain``.
    pickle_filename : str
        Relative path and filename of the backup pickle file.
    timestep : int
        The age of the ``brain`` in discrete time stpes.
    """

    def __init__(self, num_sensors, num_actions, brain_name='test_brain'):
        """
        Configure the Brain.

        Parameters
        ----------
        num_sensors, num_actions : int
            Value for ``self.num_sensors`` and ``self.num_actions``.
        brain_name : str, optional
            A unique identifying name for the brain. 
            The default is 'test_brain'.
        """
        # Include two extra sensors. These are for explicitly sensing reward
        # and punshment.
        self.num_sensors = num_sensors + 2
        # Always include an extra action. The last is the 'do nothing' action.
        self.num_actions = num_actions + 1
        self.backup_interval = 1e5
        self.name = brain_name
        self.log_dir = os.path.normpath(os.path.join(mod_path, '..', 'log'))
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.pickle_filename = os.path.join(self.log_dir, 
                                            '{0}.pickle'.format(brain_name))

        self.cortex = cortex.Cortex(self.num_sensors)
        self.num_features = self.cortex.size
        self.timestep = 0

        # Initialize all the components of the ``brain``.
        self.amygdala = amygdala.Amygdala(self.num_features)
        self.cerebellum = cerebellum.Cerebellum(self.num_features, 
                                                self.num_actions)
        self.ganglia = ganglia.Ganglia(self.num_features, self.num_actions)

    def sense_act_learn(self, sensors, reward):
        """
        Take sensor and reward data in and use them to choose an action.

        Parameters
        ----------
        sensors : array of floats
            The information coming from the sensors in the world.
            The array should have ``self.num_sensors`` elements.
            Each value in the array is expected to be between 0 and 1,
            inclusive. Sensor values are interpreted as fuzzy binary 
            values, rather than continuous values. For instance,
            the ``brain`` doesn't interpret a contact sensor value of .5 
            to mean that the contact
            sensor was only weakly contacted. It interprets it
            to mean that the sensor was fully contacted for 50% of the sensing
            duration or that there is a 50% chance that the sensor was
            fully contacted during the entire sensing duration. For another
            example, a light sensor reading of zero won't be 
            interpreted as by the ``brain`` as darkness. It will just be 
            interpreted as a lack of information about the lightness.
        reward : float
            The extent to which the ``brain`` is being rewarded by the 
            world. It is expected to be between -1 and 1, inclusive.
            -1 is the worst pain ever. 1 is the most intense ecstasy 
            imaginable. 0 is neutral.

        Returns
        -------
        actions : array of floats
            The action commands that the ``brain`` is sending to the world
            to be executed. The array should have ``self.num_actions``
            elements in it. Each value should be binary: 0 and 1. This
            allows the ``brain`` to learn most effectively how to interact 
            with the world to obtain more reward. 
        """
        self.timestep += 1
        # Calcuate activities of all the features.
        features = self.cortex.featurize(sensors, reward)
         
        # Decide which actions to take.
        (decision_values, 
         feature_importance) = self.cerebellum.get_decision_values(
                features, self.amygdala.reward_by_feature, self.ganglia.goals)
        actions, goals = self.ganglia.decide(features, decision_values)
       
        # Learn from this new time step of experience.
        grow = self.cortex.learn(feature_importance)
        satisfaction = self.amygdala.learn(features, reward) 
        self.cerebellum.learn(features, actions, goals, satisfaction)

        # If the ``cortex`` just added a new level to its hierarchy, 
        # grow the rest of the components accordingly.
        if grow:
            self.amygdala.grow(self.cortex.size)
            self.cerebellum.grow(self.cortex.size)
            self.ganglia.grow(self.cortex.size)

        if (self.timestep % self.backup_interval) == 0:
            self.backup()  
        # Account for the fact that the last "do nothing" action 
        # was added by the ``brain``.   
        verbose = False
        if verbose:
            print
            print 'sensors'
            tools.format(sensors)
            print 'reward', reward
            print features
            tools.format(features)
            print 'actions'
            tools.format(actions)

        return actions[:-1]

    def visualize(self):
        """ 
        Show the current state and some history of the brain 
        
        This is typically called from a world's ``visualize`` method.
        """
        print('{0} is {1} time steps old'.format(self.name, self.timestep))

        self.amygdala.visualize(self.timestep, self.name, self.log_dir)
        #self.cerebellum.visualize(self.name, self.log_dir)
        self.ganglia.visualize(self.name, self.log_dir)
        self.cortex.visualize()
 
    def report_performance(self):
        """
        Make a report of how the brain did over its lifetime.
        
        Returns
        -------
        performance : float
            The average reward per time step collected by
            the ``brain`` over its lifetime.
        """
        performance = self.amygdala.visualize(self.timestep, 
                                              self.name, 
                                              self.log_dir)
        print('Final performance is {0:.3}'.format(performance))
        #self.backup()
        return performance

    def backup(self):
        """ 
        Archive a copy of the brain object for future use.

        Returns
        -------
        success : bool
            If the backup process completed without any problems, ``success``
            is True, otherwise it is False.
        """
        success = False
        try:
            with open(self.pickle_filename, 'wb') as brain_data:
                pickle.dump(self, brain_data)
            # Save a second copy. If you only save one, and the user
            # happens to ^C out of the program while it is being saved,
            # the file becomes corrupted, and all the learning that the
            # ``brain`` did is lost.
            with open('{0}.bak'.format(self.pickle_filename), 
                      'wb') as brain_data_bak:
                pickle.dump(self, brain_data_bak)
        except IOError as err:
            print('File error: {0} encountered while saving brain data'.
                    format(err))
        except pickle.PickleError as perr: 
            print('Pickling error: {0} encountered while saving brain data'.
                    format(perr))        
        else:
            success = True
        return success
        
    def restore(self):
        """ 
        Reconstitute the brain from a previously saved brain. 

        Returns
        -------
        restored_brain : Brain
            If restoration was successful, the saved ``brain`` is returned.
            Otherwise a notification prints and a new ``brain`` is returned.
        """
        restored_brain = self
        try:
            with open(self.pickle_filename, 'rb') as brain_data:
                loaded_brain = pickle.load(brain_data)
            """
            Compare the number of channels in the restored brain with 
            those in the already initialized brain. If it matches, 
            accept the brain. If it doesn't,
            print a message, and keep the just-initialized brain.
            Sometimes the pickle file is corrputed. When this is the case
            you can manually overwrite it by removing the .bak from the 
            .pickle.bak file. Then you can restore from the backup pickle.
            """
            if ((loaded_brain.num_sensors == self.num_sensors) and 
                (loaded_brain.num_actions == self.num_actions)):
                print('Brain restored at timestep {0} from {1}'.format(
                        str(loaded_brain.timestep), self.pickle_filename))
                restored_brain = loaded_brain
            else:
                print('The brain {0} does not have the same number'.format(
                        self.pickle_filename)) 
                print('of input and output elements as the world.')
                print('Creating a new brain from scratch.')
        except IOError:
            print('Couldn\'t open {0} for loading'.format(
                    self.pickle_filename))
        except pickle.PickleError, e:
            print('Error unpickling world: {0}'.format(e))
        return restored_brain
