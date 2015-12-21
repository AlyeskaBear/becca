""" 
The Cortex class 
"""
import numpy as np
import ziptie

class Cortex(object):
    """
    A deep feature creator. A hierarchical sparse neural network.

    The cerebral cortex is on the outside of the brain. It is the 
    easiest part of the brain to reach surgically and to insert electrodes
    into, so it is the subject of more human in vivo studies than 
    any other part of the brain. Specific regions of it tend to be active
    during specific activities, from low-level perception, to the 
    most abstract thinking and planning. In fact, popular computational 
    models of human brain function have ignored other brain areas altogther.

    The structure of the six layers of the cortex is
    roughly the same throughout the enire surface of the brain. This
    suggests that there is one universal algorithm that does 
    everything, making a tempting target for computational modelers and
    algorithm designers. But even though everyone agrees that the cortex is
    important, hardly anyone agrees on how it does what it does. Here's my 
    best guess.
    
    The cortex performs creates a hierarchy of clusters. At the lowest
    level these are clusters of simple inputs. For example a few
    retinal center-surround sensors are clustered to create a short 
    line segment. In the middle levels, these clusters are concrete but
    complex. For example, patterns of shapes, motion, color, sound, and touch
    are clustered together to create a representation of a 69 Ford Mustang.
    At the highest levels, these clusters are abstract concepts. For example,
    the representations of all the cars, trucks, planes, tractors, bicycles,
    skateboards, and hang gliders I've ever experienced would be grouped
    together with Radio Flyer wagons and the Space Shuttle to form the 
    concept of transportation.

    In BECCA, the ``cortex`` does this using multiple instances of an 
    incremental clustering algorithm, the ``ziptie``. A sequence of zipties
    form the hierarchy. 
    On each time step sensor activities are passed up the drivetrain 
    and processed into increasingly sophisticated features.

    Attributes
    ----------
    num_zipties : int
        The number of zipties that have been created. Also the length of
        ``self.ziptie``..
    size : int
        The number of inputs (cables) each ziptie can handle. 
        Also the number of features (bundles) each ziptie returns. The
        ``cortex`` fixes all these to be equal for convenience only.
        The ziptie algorithm doesn't require it. 
    zipties : list of ZipTie
        Each ``ZipTie`` instance is one level in the hierarchy of 
        the deep feature creator.
        See the documentation for the ``ZipTie`` class in 
        the ``ziptie.py`` module.
    """
    
    def __init__(self, num_sensors):
        """ 
        Initialize the ``Cortex`` with a ``ZipTie`` clustering algorithm..

        Parameters
        ----------
        num_sensors : int
            See docstring in ``brain.py``.
        """
        self.num_zipties =  1
        # Choose size to allow for sensing reward and for feature proliferation.
        self.size = 3 * num_sensors 
        first_ziptie_name = ''.join(('ziptie_', str(self.num_zipties - 1)))
        self.zipties = [ziptie.ZipTie(self.size, name=first_ziptie_name)]

    def featurize(self, sensors, reward):
        """ 
        Take in new sensor values and find the corresponding feature values.
        
        At each time step, pass the sensor values up through the 
        hierarchy of ``zipties`` and find the feature values that result.
        Also, use the new sensor values to incrementally train the 
        ``ziptie``s.

        Parameters
        ----------
        sensors : array of floats
            The most recent set of sensor activities (values) 
            returned by the world.
            These should all be between 0 and 1.
        reward : float
            The most recent reward value returned by the world

        Returns
        -------
        feature_activities : array of floats
            The current activities (values) of all the features 
        ziptie_added : bool
            If True, this indicates that another ``ZipTie`` was added to
            ``self.zipties``, another level added to the hierarchy.
        -------

        """
        # Explicitly sense reward. This makes it a sensed value
        # and helps the brain to make plans to reach it.
        negative_reward = np.maximum(-reward, 0)
        positive_reward = np.maximum(reward, 0) 
        self.sensors = np.zeros(sensors.size + 2)
        self.sensors[:sensors.size] = sensors
        self.sensors[-2] = negative_reward
        self.sensors[-1] = positive_reward
        cable_activities = self.sensors.copy()
        # Step through the ``ZipTie``s in ``self.zipties`` one at a 
        # time, starting at the bottom-most and proceeding upward.
        # The output features (bundles) of each are the input features
        # (cables) of the next.
        for level, tie in enumerate(self.zipties):
            cable_activities = tie.featurize(cable_activities) 

        # Build the full feature activities array by appending the
        # feature inputs from all the ``ZipTie``s.
        num_features = self.size * len(self.zipties)
        feature_activities = np.zeros(num_features)
        for (ziptie_index, tie) in enumerate(self.zipties):
            self.level_scale = .6
            scale = self.level_scale ** (len(self.zipties) - ziptie_index - 1)
            start_index = self.size * ziptie_index
            end_index = self.size * (ziptie_index + 1)
            feature_activities[start_index: end_index] = (
                    tie.cable_activities.copy() * scale)

        #print feature_activities
        return feature_activities

    def learn(self, feature_importance):
        """ 
        Take in new sensor values and find the corresponding feature values.
        
        At each time step, pass the sensor values up through the 
        hierarchy of ``zipties`` and find the feature values that result.
        Also, use the new sensor values to incrementally train the 
        ``ziptie``s.

        Parameters
        ----------
        feature_importance : array of floats

        Returns
        -------
        ziptie_added : bool
            If True, this indicates that another ``ZipTie`` was added to
            ``self.zipties``, another level added to the hierarchy.
        -------

        """
        # debug: Don't build features
        skip = False
        if skip:
            feature_activities = np.zeros(self.size)
            feature_activities[:self.sensors.size] = self.sensors
            return False

        # Step through the ``ZipTie``s in ``self.zipties`` one at a 
        # time, starting at the bottom-most and proceeding upward.
        # The output features (bundles) of each are the input features
        # (cables) of the next.
        sparse_activities = self.sensors.copy()
        for level, tie in enumerate(self.zipties):
            importance = feature_importance[self.size * level:
                                            self.size * (level + 1)]
            sparse_activities = tie.sparse_featurize(sparse_activities,
                                                     importance)
            tie.learn() 

        # If the top ziptie has created its first two bundles,
        # create a new one on top of that.
        ziptie_added = False
        if tie.num_bundles > 1:
            self.add_ziptie()
            ziptie_added = True
            sparse_activities = self.zipties[-1].sparse_featurize(
                    sparse_activities) 

        # Build the full feature activities array by appending the
        # feature inputs from all the ``ZipTie``s.
        num_features = self.size * len(self.zipties)
        sparse_feature_activities = np.zeros(num_features)
        for (ziptie_index, tie) in enumerate(self.zipties):
            start_index = self.size * ziptie_index
            end_index = self.size * (ziptie_index + 1)
            sparse_feature_activities[start_index: end_index] = (
                    tie.cable_activities.copy())
        #print 's', sparse_feature_activities
        return ziptie_added
        # Debug: I don't think that any other parts of BECCA need the sparse
        # feature activities yet, but I'll keep them available
        # in case they become useful for cognition.
        #return sparse_feature_activities, ziptie_added

    def map_index(self, index):
        """ 
        Find the ziptie and cable index that match a feature index.

        Parameters
        ----------
        index : int
            The feature index to identify.

        Returns
        -------
        cable_index : int
            The index of the input to the ``ZipTie`` in ``self.zipties``
            asociated with the feature index.
        ziptie_index : int
            The index of the ``ZipTie`` in ``self.zipties`` associated
            with the feature index.
        """
        if index is None:
            return None, None
        # else
        ziptie_index = int(np.floor(index / self.size ))
        cable_index = index - ziptie_index * self.size 
        return ziptie_index, cable_index

    def add_ziptie(self):
        """ 
        When the last ziptie creates its first bundle, add a ziptie.
        """
        next_ziptie_name = ''.join(('ziptie_', str(self.num_zipties)))
        self.zipties.append(ziptie.ZipTie(self.size,
                                          name=next_ziptie_name, 
                                          level=self.num_zipties))
        print "Added ziptie", self.num_zipties
        self.num_zipties +=  1

    def get_index_projections(self, to_screen=False):
        """
        Get representations of all the bundles in each ziptie 
        
        Every feature is projected down through its own ziptie and
        the zipties below it until its cable_contributions on sensor inputs 
        and actions is obtained. This is a way to represent the
        receptive field of each feature.

        The workhorse for this is another method, ``_get_index_projections``.

        Parameters
        ----------
        to_screen : bool
            If True, print a text version of the projections to the console.
            Default is False.

        Returns
        -------
        all_projections : list of array of ints
            All the sensor contributions for each feature. 
            ``all_projections[i]`` gives an array of sensor indices that 
            contribute to that feature. Unlike the features array elsewhere
            in the BECCA code, this does not include sensors 
        all_bundle_activities : list of floats
            The current activities of each feature (bundle).
            Unlike the features array elsewhere
            in the BECCA code, this does not include sensors 
        """
        all_projections = []
        all_bundle_activities = []

        # Cycle up through the lsit of ``ZipTie``s.
        for ziptie_index in range(len(self.zipties)):
            ziptie_projections = []
            ziptie_bundle_activities = []
            num_bundles = self.zipties[ziptie_index].max_num_bundles
            # Cycle through each ``ZipTie``'s array of bundles (features).
            for bundle_index in range(num_bundles):    
                bundles = np.zeros((num_bundles, 1))
                bundles[bundle_index, 0] = 1.
                # For each bundle (feature), find all the cables (sensors) 
                # that contribute to it.
                cable_contributions = self._get_index_projection(
                        ziptie_index,bundles)
                if np.nonzero(cable_contributions)[0].size > 0:
                    ziptie_projections.append(cable_contributions)
                    ziptie_bundle_activities.append(
                            self.zipties[ziptie_index].
                            bundle_activities[bundle_index])
                    # Display the cable_contributions in text form if desired
                    if to_screen:
                        print ' '.join(['cable_contributions', 
                                        self.zipties[ziptie_index].name,
                                        'feature', str(bundle_index), 
                                        'cables', str(list(
                                        np.nonzero(cable_contributions)[0]))])
                        #for i in range(cable_contributions.size):
                        #    print np.nonzero(cable_contributions)[0][
                        #            np.where(np.nonzero(
                        #            cable_contributions)[1] == i)]
            # Assemble the final list.
            if len(ziptie_projections) > 0:
                all_projections.append(ziptie_projections)
                all_bundle_activities.append(ziptie_bundle_activities)
        return (all_projections, all_bundle_activities)

    def _get_index_projection(self, ziptie_index, bundles):
        """
        Get the sensor contributions for bundles.
        
        Recursively project bundles (features) down through ``self.zipties``
        until the bottom ``ZipTie`` is reached. 

        Return the cable_contributions in terms of basic sensor 
        inputs. 

        Parameters
        ----------
        bundles : array of floats
            The set of activities (values) of the current ``ZipTie``'s
            bundles (features).
        ziptie_index : int
            The current ``ZipTie``'s location in the list of
            ``self.zipties``. Also, its position in the hierarchy.

        Returns
        -------
        bundles : array of floats
            When a ``ZipTie`` past the end of the list is called,
            return the input argument back to the calling function.
        cable_contributions : array of floats
            Otherwise, return the original cable (sensor) indices that
            contribute to ``bundles``.
        """
        # Once the bottom of the recursion is reached, exit.
        if ziptie_index == -1:
            return bundles

        # Otherwise, continue recursing downward through the ``ZipTie``s.
        cable_contributions = np.zeros(self.zipties[ziptie_index].
                                       max_num_cables)
        # Cycle through each feature (bundle) and find the sensors that
        # contribute to it.
        for bundle_index in range(bundles.size):
            if bundles[bundle_index] > 0:
                # Call the ``get_index_projection`` method of each 
                # ``ZipTie``, not to be confused with that of the ``Cortex``.
                new_contribution = self.zipties[
                        ziptie_index].get_index_projection(bundle_index)
                cable_contributions = (np.maximum(
                        cable_contributions, new_contribution))
                
        # Recursion--Now call this method again for the next ``ZipTie``
        # lower in the hierarchy.
        cable_contributions = self._get_index_projection(ziptie_index - 1, 
                                                         cable_contributions)
        return cable_contributions

    def visualize(self):
        """
        Show the ``Cortex`` in pictures. It's really all about the ``ZipTie``s.
        """
        print 'cortex:'
        for tie in self.zipties:
            tie.visualize()
