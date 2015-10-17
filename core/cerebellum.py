"""
The Cerebellum class
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tools

class Cerebellum(object):
    """
    Predict the next time step's most likely features and actions.
    
    Like BECCA's other functional models of brain regions, the exact role
    of the cerebellum is hotly contested. It's structure is more regular 
    and has been mapped in more detail than that of any other structure
    except perhaps the hippocampus. Here's an educated guess about
    what it's doing, but consider yourself warned: It doesn't exactly match
    anyone else's guess, including those of some extremely smart people
    who have been studying it for decades. The only consolation I have
    is that the smart people disagree with each other too. 
    
    Everyone agrees that the cerebellum has a big role in the control 
    and execution of automatic movements. I hypothesize that it does
    this by predicting sensor activity and actions for the near future.
    Predicted actions become automatic reactions unless inhibited. I'm
    not the first to propose this, but this particularly formulation
    is uniquely mine. I can't blame its shortcomings on anyone else.

    Attributes
    ----------
    age : 2D array of floats
        The number of time steps since a hypothesis has been used to
        predict a feature or action. This is a float rather than an int
        because it decays, resulting in fractional values.
        The indexing scheme is similar to that of ``hypos``:
            Index 0 : which feature/action is being predicted
            Index 1 : position in the list of hypotheses
    decay_rate : float
        The fraction of recent feature and action activity that 
        decays at each time step.
    hypos, samples : 3D array of floats
        ``samples`` is a list of discrete experiences corresponding 
        to activity in each feature and action. 
        ``hypos`` is a list of hypotheses for predicting each feature 
        and action. In both cases, the actions and features are 
        concatenated into a single array, actions first. The indexing
        scheme of both is as follows:
            Index 0 : which feature/action is being predicted
            Index 1 : position in the list of samples or hypotheses
            Index 2 : which features/actions preceded the feature/action 
                    being predicted
        Values are constrained to be zero or one. This is a sparse binary
        array and eventually it will probably be re-represented in order 
        to decrease storage space and access time.
    hypos_filled, samples_filled : array of bool
        Indicates whether the sample or hypo collection 
        for each feature and action
        has been filled at least once. The array is one dimensional,
        ``num_samples`` elements long, and the index corresponds to 
        the feature index 0 from the samples or hypos array.
    hypos_next, samples_next : array of ints
        These are arrays of indices indicating which hypothesis or 
        sample should be added next. The position in the array 
        indicates the predicted feature (index 0) to which the index
        corresponds. The value of the array at that position is the
        hypothesis or sample index (index 1) that is next in line to
        be added.
    num_combos : int
        The combined number of actions and features.
    recent_combo : array of floats
        The decayed recent activities of actions and features. 
        This is a leaky integral, or filtered version, of the recent 
        feature and action history.
    num_samples : int
        The maximum number of samples that can be collected for predicting
        a given feature. For now, this number is also used as a 
        maximum on the number of hypotheses. 
    tries : 2D array of floats
        The number of times a hypothesis has been used to
        predict a feature or action. This is a float rather than an int
        because it is scaled by the quality of the current set of 
        feature activies, resulting in floating point values.
        The indexing scheme is similar to that of ``hypos``:
            Index 0 : which feature/action is being predicted
            Index 1 : position in the list of hypotheses
    wins : 2D array of floats
        The number of times a hypothesis has correclty
        predicted a feature or action. This is a float rather than an int
        because it is scaled by both how well the hypothesis matches
        the current set of feature activies and by how accurately
        it predicted the feature, resulting in floating point values.
        The indexing scheme is similar to that of ``hypos``:
            Index 0 : which feature/action is being predicted
            Index 1 : position in the list of hypotheses
    """

    def __init__(self, num_features, num_actions):
        """
        Configure the ``cerebellum``.
        
        Parameters
        ----------
        num_features, num_actions : int
            See docstring for ``brain.py.num_actions``.
        """
        self.num_features = num_features
        self.num_actions = num_actions

        self.num_combos = num_actions + num_features 
        self.num_samples = 3 * self.num_combos
        _2D_size = (self.num_combos, self.num_samples)
        self.wins = np.zeros(_2D_size)
        self.tries = np.zeros(_2D_size)
        self.age = np.ones(_2D_size)
        _3D_size = (self.num_combos, self.num_samples, self.num_combos)
        self.hypos = np.zeros(_3D_size)
        self.samples = np.zeros(_3D_size)
        self.hypos_next = np.zeros(self.num_samples, dtype=int)
        self.hypos_filled = np.zeros(self.num_samples, dtype=bool)
        self.samples_next = np.zeros(self.num_samples, dtype=int)
        self.samples_filled = np.zeros(self.num_samples, dtype=bool)
        self.recent_combo = np.zeros(self.num_combos)
        self.next_combo = np.zeros(self.num_combos)
        self.guess = np.zeros((self.num_combos, self.num_samples))
        self.similarity = np.zeros((self.num_combos, self.num_samples))
        self.decay_rate = 1.
        
        self.skip = True

    def weighted_accuracy(self, wins, tries, age):
        """
        The accuracy of the hypothesis, with confidence interval and age.

        Simple accuracy is:
                accuracy = wins / tries

        This is a frequentist estimation of the average accuracy.

        Age is used to penalize the number of wins.
                wins_penalty = log10(age)

        This is a way to bring the accuracy down for 
        hypotheses that haven't been used in a long time.
        As a result, hypotheses that have been used to make predictions 
        quite a few times will take an extremely long time to age.
         
        The final result is:
                accuracy = wins - log10(age) 
                           ----------------- 
                                tries 

        Parameters
        ----------
        wins : float or array of floats
            The number of times the hypothesis was validated.
        tries : float or array of floats
            The number of the times the hypothesis was tested.
        age : float or array of floats
            The number of time steps since the hypothesis was tested.

        Returns
        -------
        accuracy : float or array of floats
            The weighted accurcy of the hypothesis.
        """
        accuracy = (wins - np.log10(age)) / (tries + tools.EPSILON)
        # Limit accuracy to the interval between 0 and 1.
        accuracy  = np.maximum(accuracy, 0.)
        accuracy  = np.minimum(accuracy, 1.)
        return accuracy
        
    def predict(self, features, actions):
        """
        Make a prediction about the next time step.

        Parameters
        ----------
        features : array of floats
            The most recent set of feature activities.
        actions : array of floats
            The most recent set of actions selected.

        Returns
        -------
        next_features : array of floats
            The predicted probabilities of each feature being
            observed on the next time step based on ``features``.
        next_actions : array of floats
        The predicted probabilities of each action being
            chosen next based on ``features``.
        """
        if self.skip:
            next_features = np.zeros(features.size)
            next_actions = np.zeros(actions.size)
            return next_features, next_actions

        # Make a prediction for each action and feature based 
        # on the hypotheses it has accumulated.
        (num_combos, num_hypos, _) = self.hypos.shape
        last_combo = np.concatenate((actions, features))
        self.next_combo = np.zeros(num_combos)
        self.guess = np.zeros((num_combos, num_hypos))
        self.similarity = np.zeros((num_combos, num_hypos))
        for i_combo in range(num_combos):
            accuracy = self.weighted_accuracy(self.wins[i_combo,:], 
                                              self.tries[i_combo,:], 
                                              self.age[i_combo,:])
            # Cycle through all the hypotheses for this element and 
            # calculate the similarity of each to the current situation.
            for i_hypo in range(num_hypos):
                hypo = self.hypos[i_combo,i_hypo,:]
                indices = np.where(hypo > .5)[0]
                if indices.size > 0:
                    # Similarity between the current combo and a hypothesis
                    # is determined by the smallest value of the combo
                    # that falls within the hypothesis.
                    self.similarity[i_combo,i_hypo] = np.min(
                            last_combo[indices])

            # Form a prediction and choose one of the strongest.
            # The guess associated with each hypo is given by: 
            #         guess = similarity * accuracy
            self.guess[i_combo,:] = (self.similarity[i_combo,:] * 
                                          accuracy + tools.EPSILON)
            vote = self.guess[i_combo,:]
            p_vote = vote / np.sum(vote)
            i_best_hypo = np.random.choice(np.arange(vote.size), p=p_vote)
            #best_vote = vote[i_best_hypo]
            self.age[i_combo, i_best_hypo] = 1.
            self.next_combo[i_combo] = self.guess[i_combo, i_best_hypo]

        next_actions = self.next_combo[:self.num_actions]
        next_sensors = self.next_combo[self.num_actions:]
        return next_sensors, next_actions

    def learn(self, features, actions):
        """
        Update the cerebellar model of the world and its dynamics.

        Parameters
        ----------
        features : array of floats
            The current set of feature activities.
        actions : array of floats
            The set of actions chosen in response to ``features``.
        """
        if self.skip:
            return

        new_combo = np.concatenate((actions, features))
        (num_combos, num_hypos, _) = self.hypos.shape
        #total_diff = np.sum(np.abs(new_combo - self.next_combo))
        for i_combo, new_val in enumerate(new_combo):

            # Increment tries and wins for every hypo.
            for i_hypo in np.arange(num_hypos):
                similarity = self.guess[i_combo, i_hypo]
                error = np.abs(similarity - new_val)
                self.tries[i_combo, i_hypo] += similarity
                self.wins[i_combo, i_hypo] += np.minimum(similarity, new_val)

            # Check whether the error is big enough to merit 
            # refining the model.
            if np.abs(error) > np.random.random_sample():
                """
                Add a new hypothesis.
                There are lots of possible schemes for forming new 
                hypotheses. Each one introduces its own bias. For now
                I'm randomly choosing either to use a randomly selected
                sample or to use the intersection 
                of two randomly selected hypotheses.
                """
                # Pick whether to combine samples or hypotheses.
                if np.random.random_sample() < .5:
                    # Try to combine two samples.
                    # Check whether there are enough samples.
                    if self.samples_filled[i_combo]:
                        # Pick the sample.
                        i_hypo = np.random.randint(self.num_samples) 
                        self._add_hypothesis(i_combo, 
                                             self.samples[i_combo,i_hypo,:],
                                             tries=10., wins=1.)
                else:
                    # Try to combine two hypotheses.
                    # Check whether there are enough hypotheses.
                    if self.hypos_filled[i_combo]:
                        # Pick the first hypothesis.
                        i_hypo_a = np.random.randint(self.num_samples) 
                        # Pick the second hypothesis.
                        # Make sure it's distinct.
                        i_hypo_b = i_hypo_a
                        while i_hypo_a == i_hypo_b:
                            i_hypo_b = np.random.randint(self.num_samples) 
                        intersection = np.minimum(
                                self.hypos[i_combo,i_hypo_a,:],
                                self.hypos[i_combo,i_hypo_b,:])
                        total_tries = (self.tries[i_combo,i_hypo_a] + 
                                       self.tries[i_combo,i_hypo_b]) 
                        total_wins = (self.wins[i_combo,i_hypo_a] + 
                                      self.wins[i_combo,i_hypo_b]) 
                        self._add_hypothesis(i_combo, intersection, 
                                             tries=total_tries, 
                                             wins=total_wins)

                # Add a new sample.
                self._add_sample(i_combo, self.recent_combo)

        # Handle the effects of time
        self.age += 1.
        self.recent_combo *= 1. - self.decay_rate
        self.recent_combo = tools.bounded_sum([self.recent_combo, new_combo])

    def _add_sample(self, i_combo, new_sample):
        """
        Add a new sample and increment the counters accordingly.

        Treat the collection of samples that predict each feature as
        a circular buffer. Fill it from start to end, then start over
        at the beginning and do it over and over again. 
        
        Parameters:
        -----------
        i_combo : int
            The combo index of the feature or action to add a context to.
        new_sample_raw : array of floats
            The new sample to be added.
        """
        new_sample_bin = (new_sample > 
                          np.random.random_sample(new_sample.shape)
                          ).astype(float)
        i_hypo = self.samples_next[i_combo]
        self.samples[i_combo,i_hypo,:] = new_sample_bin
        
        # Update the index and wrap it if necessary. 
        i_hypo_next = i_hypo + 1
        if i_hypo_next == self.num_samples:
            i_hypo_next = 0
            self.samples_filled[i_combo] = True
        self.samples_next[i_combo] = i_hypo_next

    def _add_hypothesis(self, i_combo, new_hypo, tries, wins):
        """
        Add a new hypothesis and increment the counters accordingly.
        
        The biggest question here is which hypothesis to replace if the 
        collection is already full. That is determined by the track record
        of the hypothesis. Each starts perfect, then decays over time.
        The more times the hypothesis has been right, the slower it decays.
        
        Parameters:
        -----------
        i_combo : int
            The combo index of the feature or action to add a context to.
        new_hypo : array of floats
            The hypothesis to be added.
        tries : float
            The number of tries associated with the ```new_hypo```.
        wins : float
            The number of wins associated with the ```new_hypo```.

        """
        # If the proposed hypothesis has no features, it is empty. Ignore it.
        indices = np.where(new_hypo != 0.)[0]
        if indices.size == 0.:
            return

        # It the proposed hypo is already in the list, ignore it.
        diff = new_hypo - self.hypos[i_combo,:,:]
        mismatch = np.sum(np.abs(diff), axis=1)
        if (np.where(mismatch == 0)[0]).size > 0:
            return

        # First handle the case where the hypothesis collection 
        # has not yet been filled.
        if not self.hypos_filled[i_combo]:
            i_hypo = self.hypos_next[i_combo]

            # Update the index and wrap it if necessary. 
            i_hypo_next = i_hypo + 1
            if i_hypo_next == self.num_samples:
                i_hypo_next = 0
                self.hypos_filled[i_combo] = True
            self.hypos_next[i_combo] = i_hypo_next
        # The handle the case where the hypothesis collection 
        # has already been filled.
        else:
            # Replace the hypothesis that has the lowest low accuracy.
            accuracy = self.weighted_accuracy(self.wins[i_combo,:], 
                                              self.tries[i_combo,:], 
                                              self.age[i_combo,:])
            min_accuracy = np.min(accuracy)
            i_hypo_mins = np.where(accuracy == min_accuracy)[0]
            i_hypo = np.random.choice(i_hypo_mins)

        # The the new hypothesis with its tries, wins, and age.
        self.hypos[i_combo,i_hypo,:] = new_hypo
        self.tries[i_combo, i_hypo] = tries
        self.wins[i_combo, i_hypo] = wins
        self.age[i_combo, i_hypo] = 0.

    def visualize(self, brain_name, log_dir):
        """
        Represent the state of the ``cerebellum`` in pictures.
        
        The core of the ``cerebellum`` is the ``hypos`` array and,
        to a lesser extent, the ``samples`` array. Visualizing a 
        3D array of floats is challenging. It would probably be too much
        to show is all, but presenting a random sample is nearly 
        as effective at conveying the flavor of the full array.  

        Parameters
        ----------
        brain_name : str
            See docstring for ``brain.py``.
        log_dir : str
            See docstring for ``brain.py``.
        """

        def init_plot(fig_num, num_features):
            """
            Initialize a ``cerebellum`` one-step prediction plot.

            Parameters
            ----------
            fig_num : int
                The number that will be assigned to the figure.
            num_features : int
                The total number of features, actions and sensors both.

            Returns
            -------
            fig : Figure
                The figure that was created.
            """
            fig = plt.figure(num=fig_num)
            plt.clf()
            # Add pips for the predictor features and predicted features
            for i in np.arange(num_features):
                plt.plot(0., i, 'o', color=tools.DARK_COPPER)
                plt.plot(1., i, 'o', color=tools.DARK_COPPER)
                action_div = self.num_actions - .5
                plt.plot([-.09, 1.03], [action_div, action_div],
                         linewidth=.5, alpha=1., color=tools.COPPER)
                plt.text(-.07, action_div + .5, 'sensors', 
                         rotation='vertical',
                         verticalalignment='bottom', 
                         horizontalalignment='right')
                plt.text(-.07, action_div - .5, 'actions', 
                         rotation='vertical',
                         verticalalignment='top', 
                         horizontalalignment='right')
                # The approximate number of feature labels on the y-axis.
                num_labels = 24.
                label_step = int(np.ceil(num_combos / num_labels))
                y_offset = .2
                for i_action in np.arange(0, self.num_actions, label_step):
                    plt.text(-0.02, i_action + y_offset , str(i_action),
                             verticalalignment='center',
                             horizontalalignment='right')
                for i_sensor in np.arange(0, self.num_features, label_step):
                    plt.text(-0.02, self.num_actions + i_sensor + y_offset, 
                             str(i_sensor),
                             verticalalignment='center',
                             horizontalalignment='right')
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
                plt.gca().set_axis_bgcolor(tools.COPPER_HIGHLIGHT)
                plt.gca().set_xlim((-0.14, 1.1))
                plt.gca().set_ylim((-1., num_features))
            return fig

        def plot_row(row, target, thickness=1., alpha=.3):
            """
            Plot a single one-step prediction in the current axes.

            Parameters
            ----------
            row : array of floats
                The value of all the features in the prediction.
            target : int
                The index of the feature being predicted.
            """
            row = np.round(row).astype(np.int)
            i_features = np.where(row > 0)[0]
            if i_features.size > 0:
                i_features = i_features.tolist()
                i_features_plus = list(i_features)
                i_features_plus.append(target)
                y_junction = np.mean(np.array(i_features_plus))
                for i_feature in i_features:
                    plt.plot([0., .5], [i_feature, y_junction],
                             color=tools.DARK_COPPER, linewidth=thickness,
                             alpha=alpha, solid_capstyle='round')
                    plt.plot([1., .5], [target, y_junction],
                             color=tools.DARK_COPPER, linewidth=2.*thickness,
                             alpha=alpha, solid_capstyle='round')

        # Render a random selection of ``self.samples``
        # The approximate number of samples to show.
        samples_shape = self.samples.shape
        num_combos = samples_shape[0]
        num_samples = samples_shape[1]
        num_samples_show = 100
        total_samples = num_combos * num_samples
        frac_samples = float(num_samples_show) / float(total_samples)

        fig = init_plot(22222, num_combos)
        # Randomly select a fraction of samples to show.
        for i_combo in np.arange(num_combos):
            for i_row in np.arange(num_samples):
                if np.random.random_sample() < frac_samples:
                    plot_row(self.samples[i_combo,i_row,:], i_combo)
        
        plt.xlabel('Predictor and predicted features')
        plt.ylabel('Feature index')
        plt.title('{0} Cerebellum samples'.format(brain_name))
        fig.show()
        fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'cerebullum_samples_{0}.png'.format(brain_name)
        pathname = os.path.join(log_dir, filename)
        plt.savefig(pathname, format='png')

        # Make a movie of all the samples in the collection.
        # Render all the samples individually
        '''
        TODO: finish this and extend it to hypotheses.
        for i_combo in np.arange(num_combos):
            for i_row in np.arange(num_samples):
                fig = init_plot(22222)
                plot_row(self.samples[i_combo,i_row,:], i_combo)
        '''

        # Render a random selection of ``self.hypos``
        # The approximate number of hypos to show.
        hypos_shape = self.hypos.shape
        num_combos = hypos_shape[0]
        num_hypos = hypos_shape[1]
        num_hypos_show = 100
        total_hypos = num_combos * num_hypos
        frac_hypos = float(num_hypos_show) / float(total_hypos)

        fig = init_plot(22223, num_combos)
        # Randomly select a fraction of hypos to show.
        for i_combo in np.arange(num_combos):
            for i_row in np.arange(num_hypos):
                if np.random.random_sample() < frac_hypos:
                    hypo_tries = self.tries[i_combo,i_row] 
                    hypo_wins = self.wins[i_combo,i_row] 
                    hypo_age = self.age[i_combo,i_row]
                    accuracy = self.weighted_accuracy(hypo_wins,
                                                      hypo_tries,
                                                      hypo_age)
                    thickness = np.log(hypo_tries + 1.)
                    plot_row(self.hypos[i_combo,i_row,:], i_combo,
                             thickness=thickness, alpha=accuracy)
        
        plt.xlabel('Predictor and predicted features')
        plt.ylabel('Feature index')
        plt.title('{0} Cerebellum hypos'.format(brain_name))
        fig.show()
        fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'cerebullum_hypos_{0}.png'.format(brain_name)
        pathname = os.path.join(log_dir, filename)
        plt.savefig(pathname, format='png')
