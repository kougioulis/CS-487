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
#print(sys.getrecursionlimit())
import numpy as np


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

        ghostPositions = successorGameState.getGhostPositions()
        distancesToGhosts = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        distancesToFood = [manhattanDistance(newPos, food) for food in newFood.asList()]

        min_food_distance = 0
        if len(distancesToFood) > 0:
            min_food_distance = min(distancesToFood)

        min_ghost_distance = 0
        nearestGhost_scaredTime = 0

        if len(ghostPositions) > 0:
            min_ghost_distance = min(distancesToGhosts)
            nearestGhost_scaredTime = min(newScaredTimes)

            #when next to ghost and not scared, avoid certain death
            if nearestGhost_scaredTime == 0 and min_ghost_distance <= 1:
                return -999999
            # eat a scared ghost
            if nearestGhost_scaredTime > 0 and min_ghost_distance <= 1 :
                return 999999
    
        epsilon = 0.001
        value = 0.4*successorGameState.getScore() + 0.6/(min(distancesToFood) + epsilon)

        if nearestGhost_scaredTime > 0:
            # follow scared ghosts
            value = value - min_ghost_distance
        else:
            # avoid non-scared ghosts
            value = value + min_ghost_distance
            return value


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

class HybridAgent(MultiAgentSearchAgent):
    """
      Hybrid Agent
    """

    def getAction(self, gameState):

        #for pacman
        def max_value(gameState, depth):
            legalActions = gameState.getLegalActions(0)
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float("-inf")
            v = max(exp_value(gameState.generateSuccessor(0, action), 0 + 1, depth + 1) for action in legalActions)
            return v

        #for all ghosts
        def exp_value(gameState, agentIndex, depth):
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)

            prob = 1.0 / len(legalActions)
            v = 0
            for action in legalActions:
                newState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    v += max_value(newState, depth) * prob
                else:
                    v += exp_value(newState, agentIndex + 1, depth) * prob
            return v

        ghostPositions = gameState.getGhostPositions()
        distanceToGhost = [util.manhattanDistance(gameState.getPacmanPosition(), ghost) for ghost in ghostPositions]


        if(min(distanceToGhost) > 4):

            curr_pos = gameState.getPacmanPosition()
            foods = gameState.getFood().asList() #list of food coordinates
            capsules = gameState.getCapsules() #list of capsule coordinates
            walls = gameState.getWalls().asList()
            #closest food coordinate
            closest_food = min(foods, key=lambda x: util.manhattanDistance(curr_pos, x))

            legal = gameState.getLegalActions(0)
            
            #return the action to the closest food dot
            print("Closest food pellet: ", str(closest_food))
            print("Legal actions: ", legal)

            #use Breadth-First Search to find the path to the closest food
            startingNode = curr_pos 
            queue = util.Queue()
            visited =  set() #use dictionary for faster lookup

            queue.push((startingNode, []))

            print("Current position: ", str(curr_pos))

            if curr_pos == closest_food:
                return []

            #return the first action that leads to the closest food
            while not queue.isEmpty():
                node, path = queue.pop()
                if node not in visited:
                    visited.add(node)
                    if node == closest_food:
                        print("Path to follow:", path)
                        return path[0]
                    for action in legal:
                        if action == Directions.STOP:
                            continue

                        #get coordinates of the next nodes in the path
                        if action == Directions.NORTH:
                            successor = (node[0], node[1] + 1)
                        elif action == Directions.SOUTH:
                            successor = (node[0], node[1] - 1)
                        elif action == Directions.WEST:
                            successor = (node[0] - 1, node[1])
                        elif action == Directions.EAST:
                            successor = (node[0] + 1, node[1])
                        elif action == Directions.STOP:
                            successor = node
                        #if none of the above, then the action is not valid
                        else:
                            continue
                        if successor not in visited and successor not in walls:
                            queue.push((successor, path + [action]))
            
            #handling illegal moves if the closest food is not reachable by becoming a right turn or left turn reflex agent
            if not gameState.hasFood(gameState.getPacmanPosition()[0] + 1, gameState.getPacmanPosition()[1]) and gameState.hasFood(gameState.getPacmanPosition()[0] - 1, gameState.getPacmanPosition()[1]):
                legal = gameState.getLegalActions(0)
                current = gameState.getPacmanState().configuration.direction
                if current == Directions.STOP: current = Directions.NORTH
                left = Directions.LEFT[current]
                if left in legal: return left
                if current in legal: return current
                if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
                if Directions.LEFT[left] in legal: return Directions.LEFT[left]
                return Directions.STOP
            elif not gameState.hasFood(gameState.getPacmanPosition()[0] - 1, gameState.getPacmanPosition()[1]) and gameState.hasFood(gameState.getPacmanPosition()[0] + 1, gameState.getPacmanPosition()[1]):
                legal = gameState.getLegalActions(0)
                current = gameState.getPacmanState().configuration.direction
                if current == Directions.STOP: current = Directions.NORTH
                right = Directions.RIGHT[current]
                if right in legal: return right
                if current in legal: return current
                if Directions.LEFT[current] in legal: return Directions.LEFT[current]
                if Directions.RIGHT[right] in legal: return Directions.RIGHT[right]
                return Directions.STOP
            else:
                if random.uniform(0, 1) < 0.5: 
                    legal = gameState.getLegalActions(0)
                    current = gameState.getPacmanState().configuration.direction
                    if current == Directions.STOP: current = Directions.NORTH
                    right = Directions.RIGHT[current]
                    if right in legal: return right
                    if current in legal: return current
                    if Directions.LEFT[current] in legal: return Directions.LEFT[current]
                    if Directions.RIGHT[right] in legal: return Directions.RIGHT[right]
                    return Directions.STOP
                else:
                    legal = gameState.getLegalActions(0)
                    current = gameState.getPacmanState().configuration.direction
                    if current == Directions.STOP: current = Directions.NORTH
                    left = Directions.LEFT[current]
                    if left in legal: return left
                    if current in legal: return current
                    if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
                    if Directions.LEFT[left] in legal: return Directions.LEFT[left]
                    return Directions.STOP
    
        #if a ghost is close, death is near so use Expectimax
        else:
            best_action = None
            legalActions = gameState.getLegalActions()
            best_action = max(legalActions, key=lambda action: exp_value(gameState.generateSuccessor(0, action), 1, 1))
            print("A ghost is close! Performing Expectimax at location:", gameState.getPacmanPosition())
            return best_action

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