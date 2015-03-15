""" 
the Drivetrain class 
"""
import numpy as np
import ziptie

class Drivetrain(object):
    """
    The collection of zipties that form the backbone of the agent

    A drivetrain contains a hierarchical series of zipties. 
    On each time step sensor activities are passed up the drivetrain 
    and processed into increasingly sophisticated features.

    '''
    A drivetrain contains a hierarchical series of zipties. 
    The drivetrain performs two functions, 
    1) a step_up 
    where sensor activities are passed up the drivetrain and processed 
    into increasing abstract feature activities
    2) a step_down 
    where goals are passed down the drivetrain and processed 
    into increasingly concrete actions.
    '''
    """
    def __init__(self, min_cables):
        """ 
        Initialize the drivetrain.

        min_cables is the minimum number of cables that a ziptie in the
        drivetrain should be able to accomodate.
        """
        self.num_zipties =  1
        self.min_cables = min_cables
        first_ziptie_name = ''.join(('ziptie_', str(self.num_zipties - 1)))
        self.zipties = [ziptie.ZipTie(self.min_cables, 
                                          name=first_ziptie_name)]
        self.cables_per_ziptie = self.zipties[0].max_num_cables
        self.bundles_per_ziptie = self.zipties[0].max_num_bundles
        self.ziptie_added = False
        #self.surprise_history = []
        self.recent_surprise_history = [0.] * 100

    def step_up(self, action, sensors):
        """ 
        Find feature_activities that result from new cable_activities 
        """
        self.num_actions = action.size
        # debug: don't sense actions directly 
        #cable_activities = np.vstack((action, sensors))
        cable_activities = sensors

        for ziptie in self.zipties:
            cable_activities = ziptie.step_up(cable_activities) 
        # If the top ziptie has created its first two bundles,
        # create a new one on top of that.
        if ziptie.bundles_created() > 1:
            self.add_ziptie()
            cable_activities = self.zipties[-1].step_up(cable_activities) 
        # Build full feature activities array
        num_features = self.cables_per_ziptie * len(self.zipties)
        feature_activities = np.zeros((num_features , 1))
        for (ziptie_index, ziptie) in enumerate(self.zipties):
            start_index = self.cables_per_ziptie * ziptie_index
            end_index = self.cables_per_ziptie * (ziptie_index + 1)
            feature_activities[start_index: end_index] = (
                    ziptie.cable_activities.copy())
        return feature_activities

        '''
    def assign_goal(self, goal_index): 
        """
        Assign goal to the appropriate ziptie
        
        When a goal cable is selected by the hub or arborkey, 
        they doesn't know which ziptie it belongs to. 
        This method sorts that out. 
        """
        (ziptie_index, cable_index) = self.map_index(goal_index)
        # Activate the goal
        if ziptie_index is not None:
            if cable_index is not None:
                self.zipties[ziptie_index].cable_goals[cable_index] = 1.
        '''
    def map_index(self, index):
        """ 
        Find the ziptie and cable index that match a hub index 
        """
        if index is None:
            return None, None
        # else
        ziptie_index = int(np.floor(index / self.cables_per_ziptie))
        cable_index = index - ziptie_index * self.cables_per_ziptie
        return ziptie_index, cable_index
        '''
    def step_down(self):
        """ 
        Find the primitive actions driven by a set of goals 
        """
        # Propogate the deliberation_goal_votes down through the zipties
        #agent_surprise = 0.0
        cable_goals = np.zeros((self.bundles_per_ziptie, 1))
       
        for ziptie in reversed(self.zipties):
            cable_goals = ziptie.step_down(cable_goals)
            #if np.nonzero(ziptie.surprise)[0].size > 0:
            #    agent_surprise = np.sum(ziptie.surprise)
        # Tabulate and record the surprise registered in each ziptie
        #self.recent_surprise_history.pop(0)
        #self.recent_surprise_history.append(agent_surprise)
        #self.typical_surprise = np.median(np.array(
        #        self.recent_surprise_history))
        #mod_surprise = agent_surprise - self.typical_surprise
        #self.surprise_history.append(mod_surprise)
        # Report the action that resulted for the current time step.
        # Strip the actions off the cable_goals to make 
        # the current set of actions.
        action = cable_goals[:self.num_actions,:] 
        return action 
        '''
    def add_ziptie(self):
        """ 
        When the last ziptie creates its first bundle, add a ziptie 
        """
        next_ziptie_name = ''.join(('ziptie_', str(self.num_zipties)))
        self.zipties.append(ziptie.ZipTie(self.cables_per_ziptie,
                                 name=next_ziptie_name, 
                                 level=self.num_zipties))
        print "Added ziptie", self.num_zipties
        self.num_zipties +=  1
        self.ziptie_added = True

    def get_index_projections(self, to_screen=False):
        """
        Get representations of all the bundles in each ziptie 
        
        Every feature is projected down through its own ziptie and
        the zipties below it until its cable_contributions on sensor inputs 
        and actions is obtained. This is a way to represent the
        receptive field of each feature.

        Returns a list containing the cable_contributions for each feature 
        in each ziptie.
        """
        all_projections = []
        all_bundle_activities = []
        for ziptie_index in range(len(self.zipties)):
            ziptie_projections = []
            ziptie_bundle_activities = []
            num_bundles = self.zipties[ziptie_index].max_num_bundles
            for bundle_index in range(num_bundles):    
                bundles = np.zeros((num_bundles, 1))
                bundles[bundle_index, 0] = 1.
                cable_contributions = self._get_index_projection(
                        ziptie_index,bundles)
                if np.nonzero(cable_contributions)[0].size > 0:
                    ziptie_projections.append(cable_contributions)
                    ziptie_bundle_activities.append(
                            self.zipties[ziptie_index].
                            bundle_activities[bundle_index])
                    # Display the cable_contributions in text form if desired
                    if to_screen:
                        print 'cable_contributions', \
                            self.zipties[ziptie_index].name, \
                            'feature', bundle_index
                        for i in range(cable_contributions.shape[1]):
                            print np.nonzero(cable_contributions)[0][
                                    np.where(np.nonzero(
                                    cable_contributions)[1] == i)]
            if len(ziptie_projections) > 0:
                all_projections.append(ziptie_projections)
                all_bundle_activities.append(ziptie_bundle_activities)
        return (all_projections, all_bundle_activities)

    def _get_index_projection(self, ziptie_index, bundles):
        """
        Get the cable_contributions for bundles
        
        Recursively project bundles down through zipties
        until the bottom ziptie is reached. 
        ''' 
        Feature values is a 
        two-dimensional array and can contain
        several columns. Each column represents a state, and their
        order represents a temporal progression. During cable_contributions
        to the next lowest ziptie, the number of states
        increases by one. 
        '''
        Return the cable_contributions in terms of basic sensor 
        inputs. 
        if ziptie_index == -1:
            return bundles
        time_steps = bundles.shape[1] 
        cable_contributions = np.zeros(
                (self.zipties[ziptie_index].max_num_cables, time_steps * 2))
        for bundle_index in range(bundles.shape[0]):
            for time_index in range(time_steps):
                if bundles[bundle_index, time_index] > 0:
                    new_contribution = self.zipties[
                            ziptie_index].get_index_projection(bundle_index)
                    cable_contributions[:, 2*time_index: 2*time_index + 2] = ( 
                            np.maximum(cable_contributions[:, 
                            2*time_index: 2*time_index + 2], new_contribution))
        cable_contributions = self._get_index_projection(ziptie_index - 1, 
                                                         cable_contributions)
        return cable_contributions
        """
        if ziptie_index == -1:
            return bundles
        cable_contributions = np.zeros(self.zipties[ziptie_index].
                                       max_num_cables)
        for bundle_index in range(bundles.size):
            if bundles[bundle_index] > 0:
                new_contribution = self.zipties[
                        ziptie_index].get_index_projection(bundle_index)
                cable_contributions = (np.maximum(
                        cable_contributions, new_contribution))
        cable_contributions = self._get_index_projection(ziptie_index - 1, 
                                                         cable_contributions)
        return cable_contributions

    def visualize(self):
        print 'drivetrain:'
        for ziptie in self.zipties:
            ziptie.visualize()
