# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #fringe is a stack
    fringe = util.Stack()
    start = problem.getStartState()
    fringe.push([start])

    visited = []
    #visited.append(start)
    counter = 0

    while not fringe.isEmpty():
        #node = fringe.pop()
        counter += 1
        c_path = fringe.pop()


        #I am pushing the whole node into the fringe (location, direction, cost)
        #and node only want the location
        if counter == 1:
            #start doesn't have a direction or cost so...
            node = c_path[-1]
        else:
            node = c_path[-1][0]

        if problem.isGoalState(node):
            #we found a functional path!
            break
        #print("node", node)
        #print("path", c_path)
        if node not in visited:
            visited.append(node)


            for child in problem.getSuccessors(node):
                if child[0] not in visited:
                    #don't put things in the fringe that
                    #are already there, we don't want to redo spots

                    #update c_path with the new steps. but lists are muttable so
                    #have some extra steps
                    new_path = c_path.copy()
                    new_path.append(tuple(child))
                    fringe.push(new_path)
                    del new_path

    #print("end path", c_path)
    directions = [c_path[i][1] for i in range(1, len(c_path))]
    #directions is the current path with only the directions of movement
    return directions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #fringe is a stack
    fringe = util.Queue()
    start = problem.getStartState()
    fringe.push([start])

    visited = []
    visited.append(start)
    counter = 0

    while not fringe.isEmpty():
        #node = fringe.pop()
        counter += 1
        c_path = fringe.pop()

        #I am pushing the whole node into the fringe (location, direction, cost)
        #and node only want the location
        if counter == 1:
            #start doesn't have a direction or cost so...
            node = c_path[-1]
        else:
            node = c_path[-1][0]
        #print("node", node)
        #print("path", c_path)
        #print("node", node)
        if problem.isGoalState(node):
            #we found a functional path!
            break

        for child in problem.getSuccessors(node):
            #print(child)
            #print("child", child)
            if child[0] not in visited:
                #don't put things in the fringe that
                #are already there, we don't want to redo spots
                #this might not work for eat all the food
                visited.append(child[0])
                #update c_path with the new steps. but lists are muttable so
                #have some extra steps
                new_path = c_path.copy()
                new_path.append(tuple(child))
                fringe.push(new_path)
                del new_path

    directions = [c_path[i][1] for i in range(1, len(c_path))]
    #directions is the current path with only the directions of movement
    return directions


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    start = problem.getStartState()
    fringe.push(([start], 0), 0)

    visited = []
    #visited.append(start)
    counter = 0

    while not fringe.isEmpty():
        #node = fringe.pop()
        counter += 1
        c_path, prio = fringe.pop()

        #I am pushing the whole node into the fringe (location, direction, cost)
        #and node only want the location
        if counter == 1:
            #start doesn't have a direction or cost so...
            node = c_path[-1]
        else:
            node = c_path[-1][0]
        #print("node", node)
        #print("path", c_path)
        if problem.isGoalState(node):
            #we found a functional path!
            break

        if node not in visited:
            #don't put things in the fringe that
            #are already there, we don't want to redo spots
            #this might not work for eat all the food
            visited.append(node)

            for child in problem.getSuccessors(node):
                #print(child)
                if child[0] not in visited:
                    #update c_path with the new steps. but lists are muttable so
                    #have some extra steps
                    new_path = c_path.copy()
                    new_path.append(tuple(child))
                    new_prio = child[2] + prio
                    fringe.push((new_path, new_prio), new_prio)
                    del new_path

    directions = [c_path[i][1] for i in range(1, len(c_path))]
    #directions is the current path with only the directions of movement
    return directions




def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    Start = problem.getStartState()  #Get The Starting Node/state
    if problem.isGoalState(Start):     #Return instantly if ontop of goal space.
        return []

    fringe = util.PriorityQueue()   #UCS ueses a PRIOITY queue)
    vistednodes = []           #Make a list of the vistsed nodes

    Cost = heuristic(Start, problem)
    fringe.push((Start, [], Cost), Cost) #Prioqueue takes the item and prioty

    while not fringe.isEmpty():
        node, c_path, pathcost = fringe.pop()



        if node not in vistednodes:         #check if its been visted yet
            vistednodes.append(node)
                       #append to the list
            if problem.isGoalState(node):       #FOUND GOAL NODE! PRINT PATH
               # print(f"PATH FOUND = {path}")
               #print(heuristic(node, problem))
               break
            for child in problem.getSuccessors(node):  #Get the neighboring node tiles.
                prio = child[2] + pathcost
                hcost = heuristic(child[0], problem)
                new = prio + hcost
                directions = c_path + [child[1]]

                fringe.push((child[0], directions, prio), new)

    return c_path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
