import copy
import pickle
import os
import numpy as np

from becca.affect import Affect
from becca.preprocessor import Preprocessor
from becca.postprocessor import Postprocessor
from becca.featurizer import Featurizer
from becca.model import Model
from becca.actor import Actor
import becca_viz.viz as viz


class Brain(object):
    """
    A biologically motivated learning algorithm.

    Becca's Brain contains all of its learning algorithms,
    integrated into a single whole.

    Check out connector.py for an example for how to attach a world
    to a brain.
    """
    def __init__(self, world, config=None):
        """
        Configure the Brain.

        There are some superficial parameters that individual worlds
        might like to choose, like how often to visualize and
        how often to back things up. These can be changed by passing
        the appropriate key-value pairs in a dictionary.

        Parameters
        ----------
        world: World
            An environment with an appropriate step() function.
        config: dict
            Keys are brain parameters, values are desired values.

        Configuration parameters
        ------------------------
        backup_interval: int
            How often the brain will save a pickle backup of itself,
            in timesteps.
        debug: boolean
            Print informative error messages?
        log_directory : str
            The full path name to a directory where information and
            backups for the world can be stored and retrieved.
        n_features: int
            The limit on the number of features passed to the model.
            If this is smaller, Becca will run faster. If it is larger
            Becca will have more capacity to learn. It's an important
            input for determining performance.
        name: str
            A descriptive string identifying the brain.
        reporting_interval: int
            How often the brain will report on performance.
        restore : bool, optional
            If restore is True, try to restore the brain
            from a previously saved
            version, picking up where it left off.
            Otherwise it create a new one.
        visualize_interval: int
            The number of time steps between creating a new performance
            calculation and visualization of the brain.
        """
        defaults = {
            "backup_interval": 1e5,
            "debug": True,
            "log_directory": None,
            "n_features": None,
            "name": None,
            "reporting_interval": 1e3,
            "restore": True,
            "visualize_interval": 1e4,
        }
        if config is None:
            config = {}

        if config.get("name") is not None:
            self.name = config.get("name")
        else:
            self.name = '{0}_brain'.format(world.name)

        if config.get("debug") is not None:
            self.debug = config.get("debug")
        else:
            self.debug = defaults.get("debug")

        if config.get("log_directory") is not None:
            self.log_dir = config.get("log_directory")
        else:
            # Identify the full local path of the brain.py module.
            # This trick is used to conveniently locate
            # other Becca resources.
            module_path = os.path.dirname(os.path.abspath(__file__))
            # log_dir : str
            #     Relative path to the log directory.
            #     This is where backups
            #     and images of the brain's state and performance are kept.
            self.log_dir = os.path.normpath(
                os.path.join(module_path, 'log'))

        # Check whether the directory is already there. If not, create it.
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        # pickle_filename : str
        #     Relative path and filename of the backup pickle file.
        self.pickle_filename = os.path.join(
            self.log_dir, '{0}.pickle'.format(self.name))

        # One of the few constraints on the world is that it has to have
        # n_actions and n_sensors members.
        # n_actions: int
        #     This is the total number of action outputs that
        #     the world is expecting.
        # n_sensors: int
        #     The number of distinct sensors that the world
        #     will be passing in to the brain.
        self.n_actions = world.n_actions
        self.n_sensors = world.n_sensors

        self.timestep = 0

        if config.get("restore") is not None:
            restore_flag = config.get("restore")
        else:
            restore_flag = defaults.get("restore")
        if restore_flag:
            restored_brain = restore(self)

        if restore_flag and restored_brain is not None:
            self.timestep = restored_brain.timestep
            self.input_activities = restored_brain.input_activities
            self.actions = restored_brain.actions
            self.n_features = restored_brain.n_features
            self.postprocessor = restored_brain.postprocessor
            self.n_commands = restored_brain.n_commands
            self.commands = restored_brain.commands
            self.preprocessor = restored_brain.preprocessor
            self.affect = restored_brain.affect
            self.satisfaction = restored_brain.satisfaction
            self.featurizer = restored_brain.featurizer
            self.model = restored_brain.model
            self.actor = restored_brain.actor

        else:
            # Initialize everything.

            # The preprocessor takes raw sensors and commands and converts
            # them into discrete inputs.
            # Assume all actions are in a continuous space.
            # This means that it can be repeatedly subdivided to
            # generate actions of various magnitudes and increase control.
            self.preprocessor = Preprocessor(n_sensors=self.n_sensors)

            # The postprocessor converts actions to discretized actions
            # and back.
            self.postprocessor = Postprocessor(n_actions=self.n_actions)

            # actions: array of floats
            #     The set of actions to execute this time step.
            #     Initializing them to non-zero helps to kick start the
            #     act-sense-decide loop.
            self.actions = np.ones(self.n_actions) * .1

            self.affect = Affect()
            # satisfaction: float
            #     The level of contentment experienced by the brain.
            #     Higher contentment dampens curiosity and
            #     the drive to explore.
            self.satisfaction = 0.

            # n_commands: array of floats
            #     commands are discretized actions, suitable
            #     for use within becca. The postprocessor
            #     translates commands into actions.
            self.n_commands = self.postprocessor.n_commands
            self.commands = np.zeros(self.n_commands)

            if config.get("n_features") is not None:
                self.n_features = config.get("n_features")
            else:
                self.n_features = (2 * self.n_commands
                                   + 8 * self.n_sensors)

            self.input_activities = np.zeros(self.n_features)
            # The featurizer is an unsupervised learner that learns
            # features from the inputs.
            self.featurizer = Featurizer(
                debug=self.debug,
                n_inputs=self.n_features,
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

        # Finish with the superficial configuration.
        # This might change from session to session.
        if config.get("backup_interval") is not None:
            self.backup_interval = config.get("backup_interval")
        else:
            self.backup_interval = defaults.get("backup_interval")

        if config.get("reporting_interval") is not None:
            self.reporting_interval = config.get("reporting_interval")
        else:
            self.reporting_interval = defaults.get("reporting_interval")

        if config.get("visualize_interval") is not None:
            self.visualize_interval = config.get("visualize_interval")
        else:
            self.visualize_interval = defaults.get("visualize_interval")

        return

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

        (model_feature_activities,
            conditional_predictions,
            conditional_rewards,
            conditional_curiosities
        ) = self.model.step(feature_activities, reward)

        feature_goals, i_goal = self.actor.choose(
            feature_activities=model_feature_activities,
            conditional_predictions=conditional_predictions,
            conditional_rewards=conditional_rewards,
            conditional_curiosities=conditional_curiosities,
        )
        feature_pool_goals = self.model.update_goals(
            feature_goals, i_goal)

        debug_local = False
        if debug_local:
            rep = "Brain"
            rep += " last action: " + str(self.actions[0]) + ", "
            rep += " reward of " + str(reward) + ", "
            rep += " next sensors " + str(sensors)
            print(rep)

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

        # Report on performance.
        if self.timestep % self.reporting_interval == 0:
            self.affect.visualize(self)

        # Create visualization.
        if self.timestep % self.visualize_interval == 0:
            viz.visualize(self)

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
            print('Unknown error: {0} encountered while saving brain data'
                  .format(err))
        else:
            success = True
        return success


def restore(brain):
    """
    Reconstitute the brain from a previously saved brain.

    Parameters
    ----------
    brain: Brain
        The beginning ot a brain. It needs to at least have a name,
        for locating the filename, and number of sensors
        and actuators defined.

    Returns
    -------
    restored_brain: Brain
        If successful, the brain that was restored.
        If unsuccessful, None.
    """
    restored_brain = None
    try:
        with open(brain.pickle_filename, 'rb') as brain_data:
            loaded_brain = pickle.load(brain_data)

        # Compare the number of channels in the restored brain with
        # those in the already initialized brain. If it matches,
        # accept the brain. If it doesn't,
        # print a message, and keep the just-initialized brain.
        # Sometimes the pickle file is corrputed. When this is the case
        # you can manually overwrite it by removing the .bak from the
        # .pickle.bak file. Then you can restore from the backup pickle.
        if ((loaded_brain.n_sensors == brain.n_sensors) and
                (loaded_brain.n_actions == brain.n_actions)):
            print('Brain restored at timestep {0} from {1}'.format(
                str(loaded_brain.timestep), brain.pickle_filename))
            restored_brain = loaded_brain

        else:
            print('The brain {0} does not have the same number'.format(
                brain.pickle_filename))
            print('of sensors and actions as the world.')
            print('Creating a new brain from scratch.')
    except IOError:
        print('Couldn\'t open {0} for loading'
              .format(brain.pickle_filename))
    except pickle.PickleError:
        print('Error unpickling world')
    return restored_brain


def run(world, config=None):
    """
    Run Becca with a world.

    Connect the brain and the world together and run them for as long
    as the world dictates.

    Parameters
    ----------
    world : World
        The world that Becca will learn.
        See the world.py documentation for a full description.
    config: dict
        Keys are configurable brain parameters and values are their
        desired values. See Brain.configure() for a full list.

    Returns
    -------
    performance : float
        The performance of the brain over its lifespan, measured by the
        average reward it gathered per time step.
    """
    brain = Brain(world, config)

    # Start at a resting state.
    actions = np.zeros(world.n_actions)
    sensors, reward = world.step(actions)

    # Repeat the loop through the duration of the existence of the world:
    # sense, act, repeat.
    while world.is_alive():
        actions = brain.sense_act_learn(copy.deepcopy(sensors), reward)
        sensors, reward = world.step(copy.copy(actions))

    # Wrap up the run.
    try:
        world.close_world(brain)
    except AttributeError:
        print("Closing", world.name)

    performance = brain.report_performance()
    return performance
