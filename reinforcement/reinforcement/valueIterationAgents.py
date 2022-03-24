# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()


    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"


        for i in range(self.iterations):
            oldv = self.values.copy() #we update the value fucntion AFTER each iteration

            for state in self.mdp.getStates():
                #Q = util.Counter()
                Q = {}

                if self.mdp.isTerminal(state):
                    continue
                for action in self.mdp.getPossibleActions(state):
                    #do q value stuff here
                    #q[a] = R(s,a) + sum(P(s',a,s) * oldv[s'prime]
                    Q[action] = self.getQValue(state, action)
                    #print(action)
                oldv[state] = max(Q.values())

            self.values = oldv





    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        QValue = 0 #Initialize
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
        #sum trabsition * { reward + discount*value)
            QValue += prob * (self.mdp.getReward(state, action, nextState) + (self.discount * self.getValue(nextState)))

        return QValue





    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        #NOT CHOOSING THE RIGHT DIRECTION
        policy = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            policy[action] = self.getQValue(state,action)
            #if len(action) == 0:
               # return None
        #print(f"argmax{policy.argMax} ")
       # if (max(policy)) == None:
           # return None
       # print(f"max {max(policy)}")
        #return max(policy, default = 0)

        return policy.argMax() #argmax will return the index or None




    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
           # print(i)
            oldv = self.values.copy() #we update the value fucntion AFTER each iteration
            Q = {}
            state = states[i % len(states)]
            #we will update only one state in each iteration, as opposed to doing a batch-style update
            #In the first iteration, only update the value of the first state in the states list.
            #In the second iteration, only update the value of the second.
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                #do q value stuff here
                #q[a] = R(s,a) + sum(P(s',a,s) * oldv[s'prime]
                Q[action] = self.getQValue(state, action)
                #print(action)
            oldv[state] = max(Q.values())

            self.values = oldv
        #for state in self.mdp.getStates():


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        #print(mdp.getStates())
        #compute predecesors of all states.
        predecessors = {}
        for state in mdp.getStates():

            if not mdp.isTerminal(state):
                #terminal states don't have predecesors ...

                
                #if state not in predecessors:
                    #this adds (0,0) first instead of second
                #    predecessors[state] = set()

                for action in mdp.getPossibleActions(state):
                    for nextState, probs in mdp.getTransitionStatesAndProbs(state, action):
                        if not mdp.isTerminal(nextState) and probs != 0:
                            if nextState not in predecessors:
                                predecessors[nextState] = set()
                            #print(predecessors[nextState])

                            predecessors[nextState].add(state)
                            #i'm only adding the parent not grandparents or greats ...


        #I AM NOT CONFIDENT IN PREDECESSORS BEING RIGHT!! >>>

        #initialize an empty prio queue
        pq = util.PriorityQueue()
        for s in mdp.getStates():
            if not mdp.isTerminal(s):
                #Find the absolute value of the difference between the current value of
                # s in self.values and the highest Q-value across all possible actions from s

                Q = []
                for action in mdp.getPossibleActions(s):
                    #Q = []
                    Q += [self.computeQValueFromValues(s, action)]
                    #print("Q", Q)
                #print(Q)
                maxQ = max(Q)
                #print(Q, maxQ)
                #make diff neg
                diff = abs(maxQ - self.values[s]) * (-1)
                #Push s into the priority queue with priority -diff
                #pq is a min heap
                pq.push(s, diff)


        for iteration in range(self.iterations):
            if pq.isEmpty():
                #TERMINATE queue is empty
                break
            s = pq.pop()
            #print(s)
            if not mdp.isTerminal(s):
                #update s's value (if its not a terminal state) in self.values
                #print(self.values[s])
                Q = []
                for action in mdp.getPossibleActions(s):
                    Q += [self.computeQValueFromValues(s, action)]
                self.values[s] = max(Q)


            for p in predecessors[s]:
                #Find the absolute value of the difference between the current value of p
                #in self.values and the highest Q-value across all possible actions from p
                Q = []
                for action in mdp.getPossibleActions(p):
                    Q += [self.computeQValueFromValues(p, action)]
                maxQ = max(Q)
                diff = abs(self.values[p] - maxQ)

                if diff > self.theta:
                    #if diff > theta, push p into the priority queue with priority -diff,
                    # as long as it does not already exist in the priority queue with equal or lower priority.
                    pq.update(p, ((-1) * diff))
