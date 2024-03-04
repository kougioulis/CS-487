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
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    initial_state = problem.getStartState() #initial state
    fringe = util.Stack() #from util, depth-first search uses a stack, so that will be our fringe (per guidelines)

    visited_states = [ ] #empty list of visited states 
    #push triple of state, list containing actions and cost
    fringe.push( (initial_state, [ ], 0) )    

    current_state = initial_state; #initial condition

    while not fringe.isEmpty():

        (current_state, actions, cost) = fringe.pop() #pop the stack

        if not (current_state in visited_states): 
            visited_states.append(current_state) #add to list of viisted states 
            if problem.isGoalState(current_state): #if goal is reached return actions 
                return actions
                break
            #expand the children of the tree
            for (next_state, action, cost) in problem.getSuccessors(current_state):
                if not (next_state in visited_states):
                    fringe.push( (next_state, actions + [action], cost) ) #push to the stack
    #util.raiseNotDefined()
    
def breadthFirstSearch(problem):

    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    #same algorithm as before, except BFS uses a queue
    initial_state = problem.getStartState()
    fringe = util.Queue() #breadth-first search uses a queue

    visited_states = [ ] #empty list of visited states

    #push triple of state, list containing actions and cost to the fringe queue (root of tree)
    fringe.push( (initial_state, [ ], 0) )
    
    current_state = initial_state; #initial assignment

    while not fringe.isEmpty(): 

        (current_state, actions, cost) = fringe.pop() #dequeue

        if not current_state in visited_states:
            visited_states.append(current_state)
            if problem.isGoalState(current_state):
                return actions
                break
            #expand the children of the tree
            for next_state, action, cost in problem.getSuccessors(current_state): #children nodes on the tree
                if not (next_state in visited_states):
                    fringe.push( (next_state, actions + [action], cost) ) #enqueue
   # util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"

    initial_state = problem.getStartState() 
    fringe = util.PriorityQueue() #uniform-cost search uses a priority queue
    visited_states = [ ] #empty list of visited states
    total_cost = 0 

    #enqueue triple of state, list containing actions and cost with priority the total cost
    fringe.push( (initial_state, [ ], total_cost), total_cost )

    current_state = initial_state #initial assignment 
    while not fringe.isEmpty():
        (current_state, actions, total_cost) = fringe.pop()
        if not (current_state in visited_states):
            visited_states.append(current_state)
            if problem.isGoalState(current_state):
                return actions
                break
            #expand children
            for state, action, cost in problem.getSuccessors(current_state):
                if not (state in visited_states):
                    fringe.push((state, actions + [action], total_cost + cost), total_cost + cost) #enqueue
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    initial_state = problem.getStartState() 
    fringe = util.PriorityQueue() 
    visited_states = [ ]
    total_cost = 0 
    heuristic_cost = 0

    #push to the priority queue triple of state, list containing actions 
    #with priority the total h = g + h
    fringe.push( (initial_state, [ ], total_cost), heuristic_cost)
    current_state = initial_state #initial assignment 

    while not fringe.isEmpty():
        (current_state, actions, total_cost) = fringe.pop()
        if not (current_state in visited_states):
            visited_states.append(current_state)
            if problem.isGoalState(current_state):
                return actions
                break
            for state, action, cost in problem.getSuccessors(current_state):
                if not (state in visited_states):
                    heuristic_cost = total_cost + cost + heuristic(state, problem)
                    fringe.push( (state, actions + [action], total_cost + cost ), heuristic_cost)
    #util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
