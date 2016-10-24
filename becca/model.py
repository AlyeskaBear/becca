"""
The Model class.
"""

from __future__ import print_function
import os

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import becca.model_numba as nb


class Model(object):
    """
    Build a predictive model based on sequences of features, goals and reward.

    This version of Becca is model-based, meaning that it builds a
    predictive model of its world in the form of a set of sequences.
    Each sequence is of the form feature-goal-feature-reward.
    (Similar to
    state-action-reward-state, in the terminology of
    reinforcement learning. These are similar to
    state-action-reward-state-action (SARSA) tuples, as in
    Online Q-Learning using Connectionist Systems" Rummery & Niranjan (1994))

    This formulation allows for prediction. Knowing the current active features
    and recent goals, both the reward and the resulting features can be
    anticipated.

    This formulation allows for action selection. Knowing the
    current active features, goals can be chosen in order to reach a
    desired feature or to maximize reward.

    This formulation allows for planning. Feature-goal-feature tuples can
    be chained together to formulate multi-step plans while maximizing
    reward and prabability of successfully reaching the goal.
    """
    def __init__(self, num_features, brain):
        """
        Get the Model set up by allocating its variables.
        """
        # num_features : int
        #     The maximum number of features that the model can expect
        #     to incorporate. Knowing this allows the model to
        #     pre-allocate all the data structures it will need.
        self.num_features = num_features

        # previous_feature_activities,
        # feature_activities : array of floats
        #     Features are characterized by their
        #     activity, that is, their level of activation at each time step.
        #     Activity can vary between zero and one.
        self.previous_feature_activities = np.zeros(self.num_features)
        self.feature_activities = np.zeros(self.num_features)

        # previous_feature_goals,
        # feature_goals,
        # feature_goal_votes : array of floats
        #     Goals can be set for features.
        #     They are temporary incentives, used for planning and
        #     goal selection. These can vary between zero and one.
        #     Votes are used to help choose a new goal each time step.
        self.feature_goal_activities = np.zeros(self.num_features)
        self.previous_feature_goals = np.zeros(self.num_features)
        self.feature_goal_votes = np.zeros(self.num_features)

        # FAIs,
        # new_FAIs,
        # previous_FAIs,
        # FAI_base, 
        # FAI_age : array of floats
        # FGIs : array of floats
        #     Of particular interest to us are **increases** in
        #     feature activities and goals. These tend to occur at a
        #     specific point in time, so they are particularly useful
        #     in building meaningful temporal sequences.
        self.new_FAIs = np.zeros(self.num_features)
        self.FAIs = np.zeros(self.num_features)
        self.previous_FAIs = np.zeros(self.num_features)
        self.FAI_base = np.zeros(self.num_features)
        self.FAI_age = np.zeros(self.num_features)
        self.FGIs = np.zeros(self.num_features)

        # curiosities,
        # prefix_occurrences,
        # prefix_activities,
        # rewards : 2D array of floats
        # sequence_activities,
        # sequence_occurrences : 3D array of floats
        #     The properties associated with each sequence and prefix.
        #     If N is the number of features,
        #     the size of 2D arrays is N**2 and the shape of
        #     3D arrays is N**3. As a heads up, this can eat up 
        #     memory as M gets large. They are indexed as follows:
        #         index 0 : feature_1 (past)
        #         index 1 : feature_goal
        #         index 2 : feature_2 (future)
        #     The prefix arrays can be 2D because they lack
        #     information about the result.
        _2D_size = (self.num_features, self.num_features)
        _3D_size = (self.num_features, self.num_features, self.num_features)
        # Making believe that everything has occurred once in the past
        # makes it easy to believe that it might happen again in the future.
        self.prefix_activities = np.zeros(_2D_size)
        self.prefix_activities_base = np.zeros(_2D_size)
        self.prefix_activities_age = np.zeros(_2D_size)
        self.prefix_credit = np.zeros(_2D_size)
        self.prefix_credit_base = np.zeros(_2D_size)
        self.prefix_credit_age = np.zeros(_2D_size)
        self.prefix_occurrences = np.ones(_2D_size)
        self.prefix_curiosities = np.zeros(_2D_size)
        self.prefix_rewards = np.zeros(_2D_size)
        self.prefix_goal_votes = np.zeros(_2D_size)
        self.sequence_occurrences = np.ones(_3D_size)

        # TODO: add goal-agnostic sequences

        # feature_decay_rate : float
        #     The rate at which element activity decays between time steps.
        #     Decay takes five times longer for each additional level.
        self.feature_decay_rate = .999
        # prefix_decay_rate : float
        #     The rate at which prefix activity decays between time steps
        #     for the purpose of calculating reward and finding the outcome.
        #     Decay takes five times longer for each additional level.
        self.prefix_decay_rate = .999
        # reward_update_rate : float
        #     The rate at which a prefix modifies its reward estimate
        #     based on new observations.
        self.reward_update_rate = 1e-2
        # curiosity_update_rate : float
        #     One of the factors that determines he rate at which
        #     a prefix increases its curiosity.
        self.curiosity_update_rate = 1e-2
        # Include passive element-element sequences
        # jumpiness : float
        #     A factor describing the rate at which inaction causes an
        #     an increased desire to take action. After a certain length
        #     of time spent sitting still, the level will begin
        #     spontaneously taking actions.
        self.jumpiness = .01
        # time_since_goal : float
        #     The number of time steps since an action has been taken
        #     or a goal has been set.
        #     This is used together with jumpiness to adjust the threshold
        #     for accepting new proposed actions.
        self.time_since_goal = 0.

        self.set_up_visualization(brain)


    def step(self, feature_activities, live_features, reward, satisfaction):
        """
        Update the model and choose a new goal.

        Parameters
        ----------
        feature_activities : array of floats
            The current activity levels of each of the features.
        live_features : array of floats
            A binary array of all features that have every been active.
        reward : float
            The reward reported by the world during the most recent time step.
        satisfaction : float
            A filtered version of recent reward history.
        """
        print('=========================================================================================================',)
        self._update_activities(feature_activities)
        # Update sequences before prefixes.


        print('reward', reward)
        print('satisfaction', satisfaction)
        print('previous_feature_activities',
              np.where(self.previous_feature_activities > .1)[0])
        print('feature_activities', np.where(self.feature_activities > .1)[0])
        print('previousFAIs', np.where(self.previous_FAIs > .1)[0])
        print('new_FAIs', np.where(self.new_FAIs > .1)[0])
        print('FAIs', np.where(self.FAIs > .1)[0])
        print('FGIs', np.where(self.FGIs > .1)[0])
        nb.update_sequences(
            live_features,
            self.new_FAIs,
            self.prefix_activities,
            self.sequence_occurrences)
        nb.update_prefixes(
            live_features,
            self.prefix_decay_rate,
            self.previous_FAIs,
            self.FGIs,
            self.prefix_activities,
            self.prefix_activities_base,
            self.prefix_activities_age,
            self.prefix_occurrences)
        #print('',)
        #print('',)
        nb.update_rewards(
            live_features,
            self.reward_update_rate,
            reward,
            #self.prefix_activities,
            self.prefix_occurrences,
            self.prefix_credit,
            self.prefix_rewards)
        #print('',)
        #print('',)
        #print('',)
        nb.update_curiosities(
            satisfaction,
            live_features,
            self.curiosity_update_rate,
            self.prefix_occurrences,
            self.prefix_curiosities,
            self.FAIs,
            self.previous_FAIs,
            self.FGIs,
            self.feature_goal_activities)
        #print('',)
        #print('',)
        #print('',)
        self.feature_goal_votes = nb.calculate_goal_votes(
            self.num_features,
            live_features,
            #self.time_since_goal,
            #self.jumpiness,
            self.prefix_goal_votes,
            self.prefix_credit,
            self.prefix_rewards,
            self.prefix_curiosities,
            self.prefix_occurrences,
            self.sequence_occurrences,
            self.FAIs,
            self.feature_goal_activities)
        #print('',)
        #print('',)
        #print('',)
        goal_index, max_vote = self._choose_feature_goals(satisfaction)
        print('self.time_since_goal', self.time_since_goal,
              'self.jumpiness', self.jumpiness)
        print('goal_index', goal_index, 'max_vote', max_vote)
        nb.update_reward_credit(
            live_features,
            goal_index,
            max_vote,
            self.prefix_goal_votes,
            self.prefix_credit,
            self.prefix_credit_base,
            self.prefix_credit_age)
        #print('',)
        return self.feature_goal_activities


    def _update_activities(self, feature_activities):
        """
        Calculate the change in feature activities and goals.
        """
        self.previous_feature_activities = self.feature_activities  
        self.feature_activities = feature_activities

        # Do some juggling here to track the increases from both the current
        # and the previous time steps.
        self.previous_FAIs = self.FAIs.copy()
        self.new_FAIs = np.maximum(
            self.feature_activities - self.previous_feature_activities, 0.)

        exponential = True
        if exponential:
            # Do a leaky integration on feature activities to allow for some
            # time delay between a feature occurring and a goal being set.
            self.FAIs *= (1. - self.feature_decay_rate)
            self.FAIs += self.new_FAIs
            self.FAIs = np.minimum(self.FAIs, 1.)
        else:
            self.FAI_age += 1.
            i_reset = np.where(self.new_FAIs > self.previous_FAIs)
            self.FAI_base[i_reset] = self.new_FAIs[i_reset]
            self.FAI_age[i_reset] = 0.
            # Decay the feature activity increases hyperbolically.
            self.FAIs = self.FAI_base * (1. / (1. + 2. * self.FAI_age))

        self.FGIs = self.feature_goal_activities - self.previous_feature_goals
        self.FGIs = np.maximum(self.FGIs, 0.)
        return


    def _choose_feature_goals(self, satisfaction):
        """
        Using the feature_goal_votes, choose a goal.
        """
        # Choose no more than one goal at each time step.
        # Choose the feature with the largest vote. If there is a tie,
        # randomly select between them.
        self.previous_feature_goals = self.feature_goal_activities
        self.feature_goal_activities = np.zeros(self.num_features)
        max_vote = np.max(self.feature_goal_votes) 
        total_vote = max_vote + self.time_since_goal * self.jumpiness * (
            1. - satisfaction)
        # Only set a goal if the best option is better
        # than a random threshold. A goal_index of -1 indicates a non-goal.
        goal_index = -1
        if total_vote > np.random.random_sample():
            matches = np.where(self.feature_goal_votes == max_vote)[0]
            goal_index = matches[np.argmax(
                np.random.random_sample(matches.size))]
            self.feature_goal_activities[goal_index] = 1.
            self.time_since_goal = 0.
        # If no goal being set this time step.
        else:
            self.time_since_goal += 1.

        return goal_index, max_vote


    def set_up_visualization(self, brain):
        # Prepare visualization.
        plt.bone()
        # fig : matplotlib figure
        #     The figure in which a visual representation of the results
        #     will be presented.
        # ax_curiosities,
        # ax_rewards,
        # ax_ocurrences : matplotlib axes
        #     The axes in which each of these 2D arrays will be rendered
        #     as images.
        plt.figure(num=73857, figsize=(9, 9))
        plt.clf()
        self.fig, (
            (self.ax_rewards, self.ax_curiosities),
            (self.ax_activities, self.ax_occurrences)) = (
            plt.subplots(2, 2, num=73857))

        def dress_axes(ax):
            plt.sca(ax)
            # patches.Rectangle((x, y), width, height)
            ax.add_patch(patches.Rectangle(
                (-.5, brain.num_actions -.5),
                self.num_features,
                brain.num_sensors,
                facecolor='green',
                edgecolor='none',
                alpha=.16))
            ax.add_patch(patches.Rectangle(
                (brain.num_actions -.5, -.5),
                brain.num_sensors,
                self.num_features,
                facecolor='green',
                edgecolor='none',
                alpha=.16))
            ax.plot(
                [-.5, self.num_features - .5],
                [brain.num_actions - .5, brain.num_actions - .5],
                color='blue',
                linewidth=.2)
            ax.plot(
                [brain.num_actions - .5, brain.num_actions - .5],
                [-.5, self.num_features - .5],
                color='blue',
                linewidth=.2)
            ax.plot(
                [-.5, self.num_features - .5],
                [brain.num_sensors + brain.num_actions - .5,
                 brain.num_sensors + brain.num_actions - .5],
                color='blue',
                linewidth=.2)
            ax.plot(
                [brain.num_sensors + brain.num_actions - .5,
                 brain.num_sensors + brain.num_actions - .5],
                [-.5, self.num_features - .5],
                color='blue',
                linewidth=.2)
            plt.xlim([-.5, self.num_features - .5])
            plt.ylim([-.5, self.num_features - .5])
            ax.invert_yaxis()
           
        dress_axes(self.ax_rewards)
        dress_axes(self.ax_curiosities)
        dress_axes(self.ax_activities)
        dress_axes(self.ax_occurrences)
        

    def visualize(self, brain):
        """
        Make a picture of the model.

        Parameters
        ----------
        brain : Brain
            The brain that this model belongs to.
        """
        # Show prefix_rewards.
        ax = self.ax_rewards
        ax.imshow(
            self.prefix_rewards, vmin=-1., vmax=1., interpolation='nearest')
        ax.set_title('Rewards')
        ax.set_ylabel('Features')

        # Show prefix_curiosities.
        ax = self.ax_curiosities
        ax.imshow(
            self.prefix_curiosities, vmin=0., vmax=1., interpolation='nearest')
        ax.set_title('Curiosities')
        ax.set_xlabel('Goals')

        # Show prefix_activities.
        ax = self.ax_activities
        ax.imshow(
            #self.prefix_activities,
            self.prefix_credit,
            vmin=0.,
            vmax=1.,
            interpolation='nearest')
        ax.set_title('Activities')
        ax.set_xlabel('Goals')
        ax.set_ylabel('Features')

        # Show prefix_occurrences.
        ax = self.ax_occurrences
        log_occurrences = np.log10(self.prefix_occurrences + 1.)
        ax.imshow(log_occurrences, interpolation='nearest')
        ax.set_title('Occurrences, max = {0}'.format(
            int(10 ** np.max(log_occurrences))))
        ax.set_xlabel('Goals')

        self.fig.show()
        self.fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'model_history_{0}.png'.format(brain.name)
        pathname = os.path.join(brain.log_dir, filename)
        plt.savefig(pathname, format='png')
        return
