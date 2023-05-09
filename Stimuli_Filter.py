import os
import numpy as np
import pandas as pd
import cv2
import gazetools

class CognitivePOMDP():

    def __init__(self):
        self.internal_state = {}
        
    def step(self, ext, action):
        ''' Define the cognitive architecture.'''
        self._update_state_with_action(action)
        response = self._get_response()
        external_state, done = ext.external_env(response)
        stimulus, stimulus_std = self._get_stimulus(ext.external_state)
        self._update_state_with_stimulus(stimulus, stimulus_std)
        obs = self._get_obs()
        reward = self._get_reward()
        return obs, reward, done

class GazeTheory(CognitivePOMDP):

    def __init__(self):
        ''' Initialise the theoretically motivated parameters.'''
        # weight eye movement noise with distance of saccade
        self.oculamotor_noise_weight = 0.01
        # weight noise with eccentricity
        self.stimulus_noise_weight = 0.09
        # step_cost for the reward function
        self.step_cost = -1
        # super.__init__()

    def reset_internal_env(self, external_state):
        ''' The internal state includes the fixation location, the latest estimate of 
        the target location and the target uncertainty. Assumes that there is no 
        uncertainty in the fixation location.
        Assumes that width is known. All numbers are on scale -1 to 1.
        The target_std represents the strength of the prior.'''
        self.internal_state = {'fixation': np.array([-1,-1]),  
                               'target': np.array([0,0]), 
                               'target_std': 0.1,
                               'width': external_state['width'],
                               'action': np.array([-1,-1])} 
        return self._get_obs()    

    def _update_state_with_action(self, action):
        self.internal_state['action'] = action
        
    def _get_response(self):
        ''' Take an action and add noise.'''
        # !!!! should take internal_state as parameter
        move_distance = gazetools.get_distance( self.internal_state['fixation'], 
                                     self.internal_state['action'] )
        
        ocularmotor_noise = np.random.normal(0, self.oculamotor_noise_weight * move_distance, 
                                        self.internal_state['action'].shape)
        # response is action plus noise
        response = self.internal_state['action'] + ocularmotor_noise
        
        # update the ocularmotor state (internal)
        self.internal_state['fixation'] = response
        
        # make an adjustment if response is out of range. 
        response = np.clip(response,-1,1)
        return response
    
    def _get_stimulus(self, external_state):
        ''' define a psychologically plausible stimulus function in which acuity 
        falls off with eccentricity.''' 
        eccentricity = gazetools.get_distance( external_state['target'], external_state['fixation'] )
        stm_std = self.stimulus_noise_weight * eccentricity
        stimulus_noise = np.random.normal(0, stm_std, 
                                         external_state['target'].shape)
        # stimulus is the external target location plus noise
        stm = external_state['target'] + stimulus_noise
        return stm, stm_std

    
    def _update_state_with_stimulus(self, stimulus, stimulus_std):
        posterior, posterior_std = self.bayes_update(stimulus, 
                                                     stimulus_std, 
                                                     self.internal_state['target'],
                                                     self.internal_state['target_std'])
        self.internal_state['target'] = posterior
        self.internal_state['target_std'] = posterior_std

    def bayes_update(self, stimulus, stimulus_std, belief, belief_std):
        ''' A Bayes optimal function that integrates multiple stimului.
        The belief is the prior.'''
        z1, sigma1 = stimulus, stimulus_std
        z2, sigma2 = belief, belief_std
        w1 = sigma2**2 / (sigma1**2 + sigma2**2)
        w2 = sigma1**2 / (sigma1**2 + sigma2**2)
        posterior = w1*z1 + w2*z2
        posterior_std = np.sqrt( (sigma1**2 * sigma2**2)/(sigma1**2 + sigma2**2) )
        return posterior, posterior_std
    
    def _get_obs(self):
        # the Bayesian posterior has already been calculated so just return it.
        # also return the target_std so that the controller knows the uncertainty 
        # of the observation.
        #return self.internal_state['target']
        return np.array([self.internal_state['target'][0],
                        self.internal_state['target'][1],
                        self.internal_state['target_std']])
    
    def _get_reward(self):
        distance = gazetools.get_distance(self.internal_state['fixation'], 
                                self.internal_state['target'])
        
        if distance < self.internal_state['width'] / 2:
            reward = 0
        else:
            reward = -distance # a much better model of the psychological reward function is possible.
            
        return reward