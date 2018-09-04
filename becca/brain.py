import pickle
import os
import numpy as np

from becca.affect import Affect
from becca.preprocessor import Preprocessor
from becca.postprocessor import Postprocessor
from becca.featurizer import Featurizer
from becca.model import Model
from becca.actor import Actor


class Brain(object):
    """
    A biologically motivated learning algorithm.

    Becca's Brain contains all of its learning algorithms,
    integrated into a single whole.

    Check out connector.py for an example for how to attach a world
    to a brain.
    """
    def __init__(
        self,
        backup_interval=2**20,
        brain_name='test_brain',
        debug=True,
        log_directory=None,
        n_actions=4,
        n_features=64,
        n_sensors=4,
        timestep=0,
        visualize_interval=2**18,
    ):
        """
        Configure the Brain.

        Parameters
        ----------
        backup_interval: int
            How often the brain will save a pickle backup of itself,
            in timesteps.
        brain_name: str
            A descriptive string identifying the brain.
        debug: boolean
            Print informative error messages?
        log_directory : str
            The full path name to a directory where information and
            backups for the world can be stored and retrieved.
        n_actions: int
            This is the total number of action outputs that
            the world is expecting.
        n_sensors: int
            The number of distinct sensors that the world will be passing in
            to the brain.
        n_features: int
            The limit on the number of features passed to the model.
            If this is smaller, Becca will run faster. If it is larger
            Becca will have more capacity to learn. It's an important
            input for determining performance.
        timestep: int
            The age of the brain in discrete time steps.
        visualize_interval: int
            How often to visualize the world, in time steps.
        """
        self.debug = debug
        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.n_features = np.maximum(
            n_features, self.n_actions + 4 * self.n_sensors)

        self.input_activities = np.zeros(self.n_features)

        # actions: array of floats
        #     The set of actions to execute this time step.
        #     Initializing them to non-zero helps to kick start the
        #     act-sense-decide loop.
        self.actions = np.ones(self.n_actions) * .1

        # The postprocessor converts actions to discretized actions
        # and back.
        self.postprocessor = Postprocessor(n_actions=self.n_actions)

        # n_commands: array of floats
        #     commands are discretized actions, suitable for use within
        #     becca. The postprocessor translates commands into actions.
        self.n_commands = self.postprocessor.n_commands
        self.commands = np.zeros(self.n_commands)

        # The preprocessor takes raw sensors and commands and converts
        # them into discrete inputs.
        # Assume all actions are in a continuous space.
        # This means that it can be repeatedly subdivided to
        # generate actions of various magnitudes and increase control.
        self.preprocessor = Preprocessor(
            # n_commands=self.n_commands,
            n_sensors=self.n_sensors,
        )

        self.affect = Affect()
        # satisfaction: float
        #     The level of contentment experienced by the brain.
        #     Higher contentment dampens curiosity and the drive to explore.
        self.satisfaction = 0.

        # The featurizer is an unsupervised learner that learns
        # features from the inputs.
        self.featurizer = Featurizer(
            debug=self.debug,
            n_inputs=self.n_features,
            threshold=1e3,
        )
        # The model builds sequences of features and goals and reward
        # for making predictions about its world.
        self.model = Model(
            brain=self,
            debug=self.debug,
            n_features=self.n_features,
        )

        # The actor takes conditional predictions from the model and
        # uses them to choose new goals.
        self.actor = Actor(self.n_features, self)

        self.timestep = timestep
        self.visualize_interval = visualize_interval
        self.backup_interval = backup_interval
        self.name = brain_name

        if log_directory:
            self.log_dir = log_directory
        else:
            # Identify the full local path of the brain.py module.
            # This trick is used to conveniently locate other Becca resources.
            module_path = os.path.dirname(os.path.abspath(__file__))
            # log_dir : str
            #     Relative path to the log directory. This is where backups
            #     and images of the brain's state and performance are kept.
            self.log_dir = os.path.normpath(os.path.join(module_path, 'log'))

        # Check whether the directory is already there. If not, create it.
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        # pickle_filename : str
        #     Relative path and filename of the backup pickle file.
        self.pickle_filename = os.path.join(
            self.log_dir, '{0}.pickle'.format(brain_name))

    def sense_act_learn(self, sensors, reward):
        """
        Take sensor and reward data in and use them to choose an action.

        Parameters
        ----------
        sensors : array of floats
            The information coming from the sensors in the world.
            The array should have self.n_sensors inputs.
            Whatever the low and high value of each sensor, its value
            will be rescaled to fall between 0 and 1.
            Sensor values are interpreted as fuzzy binary
            values, rather than continuous values. For instance,
            the brain doesn't interpret a contact sensor value of .5
            to mean that the contact
            sensor was only weakly contacted. It interprets it
            to mean that the sensor was fully contacted for
            50% of the sensing
            duration or that there is a 50% chance that the sensor was
            fully contacted during the entire sensing duration. For another
            example, a light sensor reading of zero won't be
            interpreted as by the brain as darkness. It will just be
            interpreted as a lack of information about the lightness.
        reward : float
            The extent to which the brain is being rewarded by the
            world. It is expected to be between -1 and 1, inclusive.
            -1 is the worst pain ever. 1 is the most intense ecstasy
            imaginable. 0 is neutral.

        Returns
        -------
        actions : array of floats
            The action commands that the brain is sending to the world
            to be executed. The array should have self.n_actions
            inputs in it. Each value should be binary: 0 and 1. This
            allows the brain to learn most effectively how to interact
            with the world to obtain more reward.
        """
        self.timestep += 1

        # Calculate the "mood" of the agent.
        self.satisfaction = self.affect.update(reward)

        # Calculate new activities in a bottom-up pass.
        input_activities = self.preprocessor.convert_to_inputs(sensors)
        feature_activities = self.featurizer.featurize(
            np.concatenate((self.postprocessor.consolidated_commands,
                            input_activities)))
        (conditional_predictions,
            conditional_rewards,
            conditional_curiosities) = self.model.step(
            feature_activities, reward)
        feature_goals, i_goal = self.actor.choose(
            conditional_predictions=conditional_predictions,
            conditional_rewards=conditional_rewards,
            conditional_curiosities=conditional_curiosities,
        )
        feature_pool_goals = self.model.update_goals(feature_goals, i_goal)

        # Pass goals back down.
        input_goals = self.featurizer.defeaturize(feature_pool_goals)

        # Isolate the actions from the rest of the goals.
        self.actions = (self.postprocessor.convert_to_actions(
                        input_goals[:self.n_commands]))

        # Update the inputs in a pair of top-down/bottom-up passes.
        # Top-down
        candidate_fitness = self.model.calculate_fitness()
        self.featurizer.calculate_fitness(candidate_fitness)
        # Bottom-up
        candidate_resets = self.featurizer.update_inputs()
        feature_resets = self.model.update_inputs(candidate_resets)
        self.actor.reset(feature_resets)

        # Create a set of random actions.
        # This is occasionally helpful when debugging.
        take_random_actions = False
        if take_random_actions:
            self.actions = self.random_actions()

        # Periodically back up the brain.
        if (self.timestep % self.backup_interval) == 0:
            self.backup()

        return self.actions

    def random_actions(self):
        """
        Generate a random set of actions.

        This is used for debugging. Running a world with random
        actions gives a baseline performance floor on a world.

        Returns
        -------
        actions : array of floats
            See sense_act_learn.actions.
        """
        threshold = .1 / float(self.n_actions)
        action_strength = np.random.random_sample(self.n_actions)
        actions = np.zeros(self.n_actions)
        actions[np.where(action_strength < threshold)] = 1.
        return actions

    def report_performance(self):
        """
        Make a report of how the brain did over its lifetime.

        Returns
        -------
        performance : float
            The average reward per time step collected by
            the brain over its lifetime.
        """
        performance = self.affect.visualize(self)
        return performance

    def backup(self):
        """
        Archive a copy of the brain object for future use.

        Returns
        -------
        success : bool
            If the backup process completed without any problems, success
            is True, otherwise it is False.
        """
        success = False
        try:
            with open(self.pickle_filename, 'wb') as brain_data:
                pickle.dump(self, brain_data)
            # Save a second copy. If you only save one, and the user
            # happens to ^C out of the program while it is being saved,
            # the file becomes corrupted, and all the learning that the
            # brain did is lost.
            make_second_backup = True
            if make_second_backup:
                with open('{0}.bak'.format(self.pickle_filename),
                          'wb') as brain_data_bak:
                    pickle.dump(self, brain_data_bak)
        except IOError as err:
            print('File error: {0} encountered while saving brain data'.
                  format(err))
        except pickle.PickleError as perr:
            print('Pickling error: {0} encountered while saving brain data'.
                  format(perr))
        except Exception as err:
            print('Unknown error: {0} encountered while saving brain data'.
                  format(err))
        else:
            success = True
        return success

    def restore(self):
        """
        Reconstitute the brain from a previously saved brain.

        Returns
        -------
        restored_brain : Brain
            If restoration was successful, the saved brain is returned.
            Otherwise a notification prints and a new brain is returned.
        """
        restored_brain = self
        try:
            with open(self.pickle_filename, 'rb') as brain_data:
                loaded_brain = pickle.load(brain_data)

            # Compare the number of channels in the restored brain with
            # those in the already initialized brain. If it matches,
            # accept the brain. If it doesn't,
            # print a message, and keep the just-initialized brain.
            # Sometimes the pickle file is corrputed. When this is the case
            # you can manually overwrite it by removing the .bak from the
            # .pickle.bak file. Then you can restore from the backup pickle.
            if ((loaded_brain.n_sensors == self.n_sensors) and
                    (loaded_brain.n_actions == self.n_actions)):
                print('Brain restored at timestep {0} from {1}'.format(
                    str(loaded_brain.timestep), self.pickle_filename))
                restored_brain = loaded_brain
            else:
                print('The brain {0} does not have the same number'.format(
                    self.pickle_filename))
                print('of sensors and actions as the world.')
                print('Creating a new brain from scratch.')
        except IOError:
            print('Couldn\'t open {0} for loading'.format(self.pickle_filename))
        except pickle.PickleError:
            print('Error unpickling world')
        return restored_brain
