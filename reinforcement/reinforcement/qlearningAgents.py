# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.currentQ = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        return self.currentQ[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"







        bestQ = float('-inf')

        #action = self.getAction(state)
        #maybe loop
        for action in self.getLegalActions(state):

            if self.getQValue(state, action) > bestQ:
                bestQ = self.getQValue(state, action)

        if bestQ == float('-inf'):
            #we are a terminal state so there are no legalActions and bestQ wasnt changed
            bestQ = 0.0


        return bestQ





    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        ## FIX ME
        #you should break ties randomly for better behavior. The random.choice()
        # function will help. In a particular state, actions that your agent hasn't
        # seen before still have a Q-value, specifically a Q-value of zero, and if
        # all of the actions that your agent has seen before have a negative Q-value,
        # an unseen action may be optimal.

        bestMove = None
        bestQ = -1000000000000000000

        for action in self.getLegalActions(state):
            if self.currentQ[(state, action)] > bestQ:
                bestQ = self.currentQ[(state, action)]
                bestMove = action


        return bestMove


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)

        """
        #prob = self.
        bestMove = self.computeActionFromQValues(state)
        if util.flipCoin(self.epsilon):
            #we now return a random move
            l = self.getLegalActions(state)
            if len(l) == 0:
                #we are a terminal state
                #only have to check now cause computeAction. will return None
                bestMove = None
            else:
                bestMove = random.choice(l)
        return bestMove




    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #THIS IS CALLED
        #print(util.flipCoin(self.epsilon))
        a = self.alpha
        d = self.discount

        curQ = self.getQValue(state, action)
        nextQ = self.computeValueFromQValues(nextState)

        #print(curQ, nextQ)
        #Q(s, a) = (1 - a)*Q(s, a) + (a)(R(s, a, s') + dis * max(f(Q(s', a'), N(s', a')))

        Q = (1 - a) * curQ + a*(reward + (d * nextQ))

        self.currentQ[(state, action)] = Q


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        "*** YOUR CODE HERE ***"
        Q = 0
        #weights = self.getWeights()
        for i in self.featExtractor.getFeatures(state, action):
            feature = self.featExtractor.getFeatures(state, action)[i]
            Q += self.weights[i] * feature
        #print(Q)
        self.currentQ[(state, action)] = Q
        return Q


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #print(self.getWeights())
        #diff = (r + discount * max(Q(s', a'))) - Q(s, a)
        #print(len(self.featExtractor.getFeatures(state, action)))

        curQ = self.getQValue(state, action)
        #compute Valu is probs wrong for this ...
        #self.currentQ[(state, action)] = curQ

        nextQ = self.computeValueFromQValues(nextState)


        #diff = (reward + (self.discount * nextQ)) - curQ
        diff = reward + (self.discount * nextQ) - curQ

        #print(self.featExtractor.getFeatures(state, action))
        #weights = self.getWeights()

        for i in self.featExtractor.getFeatures(state, action):
            f = self.featExtractor.getFeatures(state, action)[i]
            self.weights[i] += self.alpha * diff * f

        #self.currentQ[(state, action)] = curQ
        #self.weights = weights



    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)
            pass
