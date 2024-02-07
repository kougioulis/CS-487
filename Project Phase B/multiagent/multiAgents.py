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

################
import sys
print(sys.getrecursionlimit()) #1000

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        distancesToGhosts = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        distancesToFood = [manhattanDistance(newPos, food) for food in newFood.asList()]

        if len(distancesToFood) == 0: #limit condition
               return 0.6*successorGameState.getScore() 
        if len(distancesToGhosts) == 0: #limit condition
                return 0.6*successorGameState.getScore()

        # weighted sum (affine transformation) of the distances to the closest food and the closest ghost
        epsilon = 0.001 #epsilon to avoid division by zero

        #TODO: Check average score for different weights
        #DONE 

        return 0.6*successorGameState.getScore() + 0.6/(min(distancesToFood) + epsilon) - 0.2/(min(distancesToGhosts) + epsilon)

        #util.raiseNotDefined()

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
        """
        "*** YOUR CODE HERE ***"

        #minimax algorithm
        #max for pacman
        def max_value(gameState, depth):
            curr_depth = depth  + 1
            if gameState.isWin() or gameState.isLose() or curr_depth == self.depth: #terminal states
                return self.evaluationFunction(gameState)

            v = float("-inf")
            for action in gameState.getLegalActions(0):
                v = max(v, min_value(gameState.generateSuccessor(0, action), 1, curr_depth))
            return v

        #min for all ghosts
        def min_value(gameState, depth, agentIndex):
            v = float("inf")
            if gameState.isWin() or gameState.isLose(): #terminal states
              return self.evaluationFunction(gameState)
              
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == (gameState.getNumAgents() - 1): #check if there is only one ghost left
                   v = min(v, max_value(gameState.generateSuccessor(agentIndex, action), depth))
                else: #all other ghosts
                   v = min(v, min_value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)) #move to the next ghost
            return v

        #find the best action
        #root node 
        v = float("-inf")
    
        best_action= '' #best action for pacman agent
        for action in gameState.getLegalActions(0):
            value = min_value(gameState.generateSuccessor(0, action), 0, 0+1)  #agentIndex = 0 
            if value > v:
                v = value
                best_action = action
        return best_action #return best root action

        #util.raiseNotDefined()

"""
*** PASS: test_cases/q3/0-lecture-6-tree.test
*** PASS: test_cases/q3/0-small-tree.test
*** PASS: test_cases/q3/1-1-minmax.test
*** PASS: test_cases/q3/1-2-minmax.test
*** PASS: test_cases/q3/1-3-minmax.test
*** PASS: test_cases/q3/1-4-minmax.test
*** PASS: test_cases/q3/1-5-minmax.test
*** PASS: test_cases/q3/1-6-minmax.test
*** PASS: test_cases/q3/1-7-minmax.test
*** PASS: test_cases/q3/1-8-minmax.test
*** PASS: test_cases/q3/2-1a-vary-depth.test
*** PASS: test_cases/q3/2-1b-vary-depth.test
*** PASS: test_cases/q3/2-2a-vary-depth.test
*** PASS: test_cases/q3/2-2b-vary-depth.test
*** PASS: test_cases/q3/2-3a-vary-depth.test
*** PASS: test_cases/q3/2-3b-vary-depth.test
*** PASS: test_cases/q3/2-4a-vary-depth.test
*** PASS: test_cases/q3/2-4b-vary-depth.test
*** PASS: test_cases/q3/2-one-ghost-3level.test
*** PASS: test_cases/q3/3-one-ghost-4level.test
*** PASS: test_cases/q3/4-two-ghosts-3level.test
*** PASS: test_cases/q3/5-two-ghosts-4level.test
*** PASS: test_cases/q3/6-tied-root.test
*** PASS: test_cases/q3/7-1a-check-depth-one-ghost.test
*** PASS: test_cases/q3/7-1b-check-depth-one-ghost.test
*** PASS: test_cases/q3/7-1c-check-depth-one-ghost.test
*** PASS: test_cases/q3/7-2a-check-depth-two-ghosts.test
*** PASS: test_cases/q3/7-2b-check-depth-two-ghosts.test
*** PASS: test_cases/q3/7-2c-check-depth-two-ghosts.test
*** Running AlphaBetaAgent on smallClassic 1 time(s).
Pacman died! Score: 84
Average Score: 84.0
Scores:        84.0
Win Rate:      0/1 (0.00)
Record:        Loss
*** Finished running AlphaBetaAgent on smallClassic after 9 seconds.
*** Won 0 out of 1 games. Average score: 84.000000 ***
*** FAIL: test_cases/q3/8-pacman-game.test
***     Bug: Wrong number of states expanded.
*** Tests failed.
*** Bug: Wrong number of states expanded.
*** Tests failed.

python3 autograder.py -q q3

"""
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        #alpha-beta pruning 
        #for pacman (agentIndex 0)
        def max_value(gameState, depth, alpha, beta):

            legal_actions = gameState.getLegalActions(0) #legal actions for pacman

            if not legal_actions or depth == self.depth: #terminal states
                return self.evaluationFunction(gameState)

            v = float("-inf")

            if depth == 0: #root (pacman)
                best_action = legal_actions[0]

            for action in legal_actions:
                successor_state = gameState.generateSuccessor(0, action)
                new_v = min_value(successor_state, depth+1, 0+1, alpha, beta) #move to the ghosts on the next depth
                
                if new_v > v:
                    v = new_v
                    if depth == 0:
                        best_action = action
                if v > beta:
                    return v
                alpha = max(alpha, v)
            if depth == 0: #root (pacman)
                return best_action #return best action of root (pacman)
            return v

        #for all ghosts
        def min_value(gameState, depth, agentIndex, alpha, beta):

            legal_actions = gameState.getLegalActions(agentIndex)

            if not legal_actions:
                return self.evaluationFunction(gameState)
              
            v = float("inf")

            for action in legal_actions:
                successor_state = gameState.generateSuccessor(agentIndex, action)

                if agentIndex == (gameState.getNumAgents() - 1): #last ghost
                   v = min(v,max_value(successor_state, depth, alpha, beta))
                else:
                   v = min(v,min_value(successor_state, depth, agentIndex + 1, alpha, beta)) #next ghost
                if v < alpha:
                    return v
                beta = min(beta, v)

            return v

        #root node 
        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        return max_value(gameState, 0, alpha, beta) #return the best action

        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
          The expectimax function returns a tuple of (actions,
        """
        "*** YOUR CODE HERE ***"

        #for pacman
        def max_value(gameState, depth):
            curr_depth = depth  + 1
            if gameState.isWin() or gameState.isLose() or curr_depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float("-inf")
            for action in gameState.getLegalActions(0):
                v = float(max(v, exp_value(gameState.generateSuccessor(0, action), 1, curr_depth)))
            return v
         
        #for all ghosts
        def exp_value(gameState, depth, agentIndex):
            v = 0
            if gameState.isWin() or gameState.isLose(): #terminal states
                return self.evaluationFunction(gameState)
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == (gameState.getNumAgents() - 1):
                    v += float(max_value(gameState.generateSuccessor(agentIndex, action), depth))
                else:
                    v += float(exp_value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)) #next ghost 
            return float(v / len(gameState.getLegalActions(agentIndex))) #expected value

        #root node
        v = float("-inf")

        best_action = ''
        for action in gameState.getLegalActions(0):
            value = float(exp_value(gameState.generateSuccessor(0, action), 0, 1))
            if value > v:
                v = value
                best_action = action
        return best_action #best action for root (pacman)

        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
                   < returns the weighted (but non-affine transformation) of all features, including capsules >
      Evaluate state by  :
            * closest food
            * food left
            * capsules left
            * distance to ghost
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    newCapsules = currentGameState.getCapsules()
    
    distancesToGhosts = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    distancesToFood = [manhattanDistance(newPos, food) for food in newFood.asList()]

    distancesToCapsules = [manhattanDistance(newPos, capsule) for capsule in newCapsules]

    #TODO: evaluate state by closest food, food left, capsules left, distance to ghost
    #DONE 
    if len(distancesToFood) == 0: #terminal state
            return currentGameState.getScore()
    if len(distancesToGhosts) == 0: #terminal state
            return currentGameState.getScore()
    if len(newCapsules) == 0: #terminal state
            return currentGameState.getScore()

    epsilon = 0.001

    if min(distancesToCapsules) < min(distancesToGhosts):
        return currentGameState.getScore() + 5/(min(distancesToFood) + epsilon) + 6/(len(newFood.asList()) + epsilon)  + 10/(min(distancesToCapsules) + epsilon)

    #weighted (but non-affine transformation) of all features, including capsules 
    return currentGameState.getScore() + 5/(min(distancesToFood) + epsilon) + 6/(len(newFood.asList()) + epsilon) + 10/(len(newCapsules) + epsilon) - 2/(min(distancesToGhosts) + epsilon)

   # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
