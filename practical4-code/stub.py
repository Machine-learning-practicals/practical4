# Imports.
import numpy as np
import numpy.random as npr
import math
import copy

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_two_state = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.bins = 10
        self.Q = np.zeros([50, 20, 20, 4]+[2])
        self.alpha = 0.1
        self.gamma = 0.95
        self.t = 1
        self.gravity_index = None
        self.epoch = 1.0

    def reset(self,epoch):
    	#self.reward_callback(self.last_reward)
        self.last_state  = None
        self.last_two_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity_index = None
        #self.alpha = 0.9/(epoch+1.0)
        #self.epoch = epoch+1.0

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        
        # make epsilon

        # ========================================
        # LAST_STATE & LAST_TWO_STATE both None
        # ========================================
        if self.last_state is None and self.last_two_state is None: # 1 round
            # new_action = int(npr.rand() < 0.1)
            new_action = 0
            self.last_action = new_action
            self.last_state  = state
            return self.last_action

        if self.last_state is not None and self.last_two_state is None: # 2 round
            # new_action = int(npr.rand() < 0.1)
            new_action = 0
            self.last_action = new_action
            # first time to compute gravity
            self.gravity_index = abs(state['monkey']['vel'] - self.last_state['monkey']['vel'])- 1
            self.last_two_state = copy.deepcopy(self.last_state)
            self.last_state = state
            return self.last_action

        # ========================================
        # LAST_STATE & LAST_TWO_STATE both are available
        # ========================================
        tree_dist_index = self.tree_dist_index(state)
        tree_top_index = self.tree_top_index(state)
        tree_bot_index = self.tree_bot_index(state)
        monkey_vel_index = self.monkey_vel_index(state)
        

        if npr.rand() > 1./self.epoch:
            new_action = np.argmax(self.Q[tree_dist_index]\
                                    [tree_top_index]\
                                    [tree_bot_index]\
                                    [self.gravity_index])
        else:
            new_action = npr.randint(0,2)

        
        # ===================
        # UPDATES
        # ===================
        self.last_action = new_action

        self.last_two_state = copy.deepcopy(self.last_state)
        self.last_state  = state
        self.t +=1
        self.alpha = 0.1/self.t    

        return self.last_action
    
    def tree_dist_index(self, state):
        # function that returns the tree_dist_index
        # assume the range(-150~500)
        tree_dist_binsize = 650.0/(50)
        tree_dist = state['tree']['dist']
        # print tree_dist
        # if tree_dist < 0: # if negative, useless
        #     tree_dist_index = 0
        # else :
        #     tree_dist_index = int(math.ceil(tree_dist/tree_dist_binsize))
        tree_dist_index = int(math.ceil((tree_dist+150)/tree_dist_binsize))

        return tree_dist_index

    def tree_top_index(self, state):
        # tree_top range (-200 ~ 400)
        tree_top = state['tree']['top'] - state['monkey']['top']
        tree_top_binsize = 600/(20)
        tree_top_index = (tree_top+200)/tree_top_binsize
        return tree_top_index

    def tree_bot_index(self, state):
        # tree_bottom ragne (-200 ~ 400)
        tree_bottom = state['monkey']['top'] - state['tree']['bot']
        tree_bottom_binsize = 600/(20)
        tree_bottom_index = (tree_bottom+200)/tree_bottom_binsize
        return tree_bottom_index

    def monkey_vel_index(self, state):
        # monkey velocity
        monkey_vel = state['monkey']['vel']
        # we assume that only deal with when vel = -40 ~ 40
        # the other 2 extreme value, we'll just put into 0th and 9th cell
        monkey_vel_binsize = 80/(20-2)
        if monkey_vel < -40:
            monkey_vel_index = 0
        elif monkey_vel > 40 :
            monkey_vel_index = 9
        else :
            monkey_vel_index = int(math.ceil((monkey_vel+40)/monkey_vel_binsize))
        return monkey_vel_index

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        
        # ========================================
        # LAST_STATE & LAST_TWO_STATE both None
        # ========================================
        if self.last_state is None and self.last_two_state is None: # 1 round
            self.last_reward = reward
            return

        if self.last_state is not None and self.last_two_state is None: # 2 round
            self.last_reward = reward
            return        

        # ========================================
        # LAST_STATE & LAST_TWO_STATE both are available
        # ========================================
        tree_dist_index = self.tree_dist_index(self.last_two_state)
        tree_top_index = self.tree_top_index(self.last_two_state)
        tree_bot_index = self.tree_bot_index(self.last_two_state)
        monkey_vel_index = self.monkey_vel_index(self.last_two_state)

        # print self.gravity_index
        # print "gravity_index", gravity_index
        # print self.last_state
        # print self.last_two_state

        # print "last sate vel", self.last_state['monkey']['vel']
        oldQ = self.Q[tree_dist_index]\
                    [tree_top_index]\
                    [tree_bot_index]\
                    [self.gravity_index]\
                    [self.last_action]

        # find out max Q(last_state, last_action)
        new_tree_dist_index = self.tree_dist_index(self.last_state)
        new_tree_top_index = self.tree_top_index(self.last_state)
        new_tree_bot_index = self.tree_bot_index(self.last_state)
        new_monkey_vel_index = self.monkey_vel_index(self.last_state)

        # print self.gravity_index
        # print self.Q[new_tree_dist_index]\
        #                                     [new_tree_top_index]\
        #                                     [new_tree_bot_index]\
        #                                     [new_monkey_vel_index]\
        #                                     [self.gravity_index]
        newQ = np.max([int(i) for i in (self.Q[new_tree_dist_index]\
                                            [new_tree_top_index]\
                                            [new_tree_bot_index]\
                                            [self.gravity_index])])
    
        self.Q[tree_dist_index]\
            [tree_top_index]\
            [tree_bot_index]\
            [self.gravity_index]\
            [self.last_action] = \
        oldQ + self.alpha*((reward + self.gamma*newQ) - oldQ)


        self.last_reward = reward
        return        


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset(ii)
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 200, 1)

	# Save history. 
	np.savetxt('score_1overT.csv',hist,delimiter=',')
	np.save('hist',np.array(hist))


