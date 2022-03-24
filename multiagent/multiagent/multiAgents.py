# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #print(type(newScaredTimes))
        #PACMAN MATH
        dist = 0
        min_dist = 100000000000
        xy1 = newPos
        unexplored = newFood.asList().copy()
        #print(unexplored)
        pacman_pos = 0

        while len(unexplored) != 0:

            for option in unexplored:
                xy2 = option

                #dist =abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
                dist = abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
                if dist < min_dist:
                    min_food = option
                    min_dist = dist
            xy1 = min_food
            #print(min_corner)
            #print(unexplored)
            unexplored.remove(min_food)
            pacman_pos += min_dist
            min_dist = 10000000000000

        #pacman_pos is a hueristic distance of how far away all the food is
        #being far away from food is bad


        #GHOST MATH
        ghostPos = successorGameState.getGhostPositions()
        xy2 = newPos
        val = 0

        for ghost in ghostPos:
            ghost = (int(ghost[0]), int(ghost[1]))
            xy1 = ghost

            dist = abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
            #if ghost is close --> freak out and return a vry neg num
            if dist <= 1:
                # WE ARE VRY CLOSE PANIC!!!!
                if newScaredTimes[0] < 2:
                    val += -10000000000
                else:
                    #we are able to eat the ghost rn so thats cool
                    #idk if this even helps makes pacman eat the food
                    val += 100000000000


        #without substracting pacman_pos it will not look for food, just tries to survive
        return successorGameState.getScore() + val - pacman_pos

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

        ### my stuff ###
        self.alpha = -100000000000000000000000000000000
        self.beta = 1000000000000000000000000000000000

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #print(gameState.getLegalActions(0))
        #print(gameState.getLegalActions(1))
        #print(gameState.generateSuccessor(1, 'North'))
        #print(gameState.getNumAgents())
        bestScore = -10000000000000000000000000
        bestMove = 'Stop'

        for move in gameState.getLegalActions(0):
            #we do the first step here cause we need to know the best move
            #that led to that path, and it is vry hard to do that through the
            #recurrsion ... i tried
            score  = self.minimize(gameState.generateSuccessor(0, move), 1, 1)
            if score > bestScore:
                bestScore = score
                bestMove = move
        #print("score", bestScore)
        #print()
        return bestMove

    ## max and min have to be separate or else we have to loop through the
    # agents which will result in weird thing happening like pac to pac paths
    # which of couse will be the maximum path ...
    def maximize(self, gameState, curDepth, agent):
        #will always be pacman so agent == 0
        #i kept agent so the calls are the same and easier to copy paste for
        #my own sanity but it is not necessay
        if agent != 0:
            print("something went vry wrong")
        #has to be before the checks cause when its not we generate sucessors on
        #the last pacman past the depth limit first
        curDepth += 1

        if gameState.isWin():
        #    print("WINNER")
            return scoreEvaluationFunction(gameState)
        elif gameState.isLose():
        #    print("LOSRE")
            return scoreEvaluationFunction(gameState)
        elif curDepth > self.depth:
            #print("DEPTHS")
            l = scoreEvaluationFunction(gameState)
            #print("depth end", l)
            return l

        score = []
        #curDepth += 1
        for move in gameState.getLegalActions(0):
            #print(move)
            #next agent will be a ghost always
            score  += [self.minimize(gameState.generateSuccessor(0, move), curDepth, 1)]
        #print("max", score)
        return max(score)


    def minimize(self, gameState, curDepth, agent):
        #we are one of the ghosts --> DONT UPDATE DEPTH
        #print("MIN")
        #print(curDepth)
        if gameState.isWin():
        #    print("WINNER")
            return scoreEvaluationFunction(gameState)
        elif gameState.isLose():
        #    print("LOSRE")
            return scoreEvaluationFunction(gameState)
        elif curDepth > self.depth:
            #print("DEPTHS")
            l = scoreEvaluationFunction(gameState)
            #print("depth end", l)
            return l
        score = []
        nextAgent = agent + 1
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            for move in gameState.getLegalActions(agent):
                #the next one is a pacman so we have to maximize
                score  += [self.maximize(gameState.generateSuccessor(agent, move), curDepth, nextAgent)]
        else:
            for move in gameState.getLegalActions(agent):
                #next one is still a ghost so we will still minimize
                #print(move)
                score  += [self.minimize(gameState.generateSuccessor(agent, move), curDepth, nextAgent)]

        #print("min", score)
        return min(score)















class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestScore = float("-inf")
        bestMove = 'Stop'
        a = float("-inf")
        b = float ("inf")

        for move in gameState.getLegalActions(0):  #Changed the start to be have depth 0 Dont know why it worked
            score  = self.ABMinimax(gameState.generateSuccessor(0, move), 0, 1, a, b)
            if score > bestScore:
                bestScore = score
                bestMove = move

            #needed to do it here TO since tech we are doing a min here.
            if bestScore > b:
                return bestScore
            a = max(a,bestScore)
        #print("score", bestScore)
        #print()
        return bestMove

    def ABMinimax(self, gameState, curDepth, Agent, a , b):
            if (gameState.isWin() or gameState.isLose() or curDepth == self.depth):
                return scoreEvaluationFunction(gameState)
            if Agent == 0:
                #if the agent is pacman go max
                return self.maximize(gameState, curDepth, Agent,a, b)
            else:
                return self.minimize(gameState, curDepth, Agent,a,b)

    def maximize(self, gameState, curDepth, agent, a, b):
        #will always be pacman so agent == 0
        #i kept agent so the calls are the same and easier to copy paste for
        #my own sanity but it is not necessay
        if agent != 0:
            print("something went vry wrong")
        #has to be before the checks cause when its not we generate sucessors on
        #the last pacman past the depth limit first
        #curDepth += 1

        #WE no Longer Need this stuff since we do a check in the helper fucntion
        """
        if gameState.isWin():
        #    print("WINNER")
            return scoreEvaluationFunction(gameState)
        elif gameState.isLose():
        #    print("LOSRE")
            return scoreEvaluationFunction(gameState)
        elif curDepth > self.depth:
            #print("DEPTHS")
            l = scoreEvaluationFunction(gameState)
            #print("depth end", l)
            return l
        """
        score = []
        v = float("-inf")
       # curDepth += 1

        for move in gameState.getLegalActions(0):
            #print(move)
            #next agent will be a ghost always

            score  += [self.ABMinimax(gameState.generateSuccessor(0, move), curDepth, 1, a, b)]
            v = max(v, score[-1])
            if v > b:
                return v
            a = max(a,v)
        return v


    def minimize(self, gameState, curDepth, agent, a, b):
        #we are one of the ghosts --> DONT UPDATE DEPTH
        #print("MIN")
        #print(curDepth)

        """
        if gameState.isWin():
        #    print("WINNER")
            return scoreEvaluationFunction(gameState)
        elif gameState.isLose():
        #    print("LOSRE")
            return scoreEvaluationFunction(gameState)
        elif curDepth > self.depth:
            #print("DEPTHS")
            l = scoreEvaluationFunction(gameState)
            #print("depth end", l)
            return l
        """
        v = float("inf")
        score = []
        nextAgent = agent + 1
        if nextAgent == gameState.getNumAgents():
            curDepth +=1    #we can set the depth here since then it goes straight into Maximize
            nextAgent = 0

        for move in gameState.getLegalActions(agent):
            #v = min(v,self.ABMinimax(gameState.generateSuccessor(agent, move), curDepth, nextAgent,a ,b))
            #the next one is a pacman so we have to maximize
            score  += [self.ABMinimax(gameState.generateSuccessor(agent, move), curDepth, nextAgent,a ,b)]
            #print("min", score)
            v = min(v, score[-1]) #get the last Part
            if v < a:
                return v
            b = min(b,v)
        return v








class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        bestScore = -10000000000000000000000000
        bestMove = 'West'

        for move in gameState.getLegalActions(0):
            #we do the first step here cause we need to know the best move
            #that led to that path, and it is vry hard to do that through the
            #recurrsion ... i tried
            score  = self.minimize(gameState.generateSuccessor(0, move), 1, 1)
            if score > bestScore:
                bestScore = score
                bestMove = move
        #print("score", bestScore)
        #print()
        return bestMove

    ## max and min have to be separate or else we have to loop through the
    # agents which will result in weird thing happening like pac to pac paths
    # which of couse will be the maximum path ...
    def maximize(self, gameState, curDepth, agent):
        #will always be pacman so agent == 0
        #i kept agent so the calls are the same and easier to copy paste for
        #my own sanity but it is not necessay
        if agent != 0:
            print("something went vry wrong")
        #has to be before the checks cause when its not we generate sucessors on
        #the last pacman past the depth limit first
        curDepth += 1

        if gameState.isWin():
        #    print("WINNER")
            return scoreEvaluationFunction(gameState)
        elif gameState.isLose():
        #    print("LOSRE")
            return scoreEvaluationFunction(gameState)
        elif curDepth > self.depth:
            #print("DEPTHS")
            l = scoreEvaluationFunction(gameState)
            #print("depth end", l)
            return l

        #score = []
        #curDepth += 1
        score = []
        numMoves = len(gameState.getLegalActions(agent))
        for move in gameState.getLegalActions(0):
            #print(move)
            #next agent will be a ghost always
            score  += [self.minimize(gameState.generateSuccessor(0, move), curDepth, 1)]
        #print("max", score)
        return max(score)


    def minimize(self, gameState, curDepth, agent):
        #we are one of the ghosts --> DONT UPDATE DEPTH
        #print("MIN")
        #print(curDepth)
        evalfunc = self.evaluationFunction

        if gameState.isWin():
        #    print("WINNER")
            return evalfunc(gameState)
        elif gameState.isLose():
        #    print("LOSRE")
            return evalfunc(gameState)
        elif curDepth > self.depth:
            #print("DEPTHS")
            l = evalfunc(gameState)
            #print("depth end", l)
            return l
        #score  = []
        probScore = 0
        nextAgent = agent + 1

        numMoves = len(gameState.getLegalActions(agent))

        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            for move in gameState.getLegalActions(agent):
                #the next one is a pacman so we have to maximize
                probScore  += (self.maximize(gameState.generateSuccessor(agent, move), curDepth, nextAgent) / numMoves)
        else:
            for move in gameState.getLegalActions(agent):
                #next one is still a ghost so we will still minimize
                #print(move)
                probScore += (self.minimize(gameState.generateSuccessor(agent, move), curDepth, nextAgent) / numMoves)

        #print("min", score)
        return probScore


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    dist = 0
    val = 0
    min_dist = -1
    ghost_dist = 0
    xy=0


    #GET PACMAN FOOD DISTANCE
    for food in currentGameState.getFood().asList():
        dist = abs(newPos[0] - food[0]) + abs(newPos[1] - food[1])
       # print(min_dist)
        if (min_dist >= dist) or (min_dist == -1):
            min_dist = dist

    #get ghosts DISTANCE

    for ghost in currentGameState.getGhostPositions():
        ghost = (int(ghost[0]), int(ghost[1]))


        ghost_dist += abs(newPos[0] - ghost[0]) + abs(newPos[1] - ghost[1])

        #if ghost is close --> freak out and return a vry neg num

        if ghost_dist <= 1:
            return float("-inf")

        if newScaredTimes[0] < 2:
                val = float("inf")
    #print(currentGameState.getScore(), (min_dist),(ghost_dist))

    return(float(currentGameState.getScore()) + (10/float(min_dist)) - (10/float(ghost_dist))) + sum(newScaredTimes)





# Abbreviation
better = betterEvaluationFunction
