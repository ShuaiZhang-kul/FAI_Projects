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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        # print(legalMoves)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        #the priority is to avoid the ghost then get more power and the last one will be get remaining score high
        ghostFactor = 5
        foodFactor = 3
        # print(f"newPos:{newPos} ,newFood:{newFood} ,newGhostStates:{newGhostStates} ,newScaredTimes:{newScaredTimes} !")
        if min(newScaredTimes) == 0: ghostFactor = -5
        distToFoods = [ util.manhattanDistance(newPos,food) for food in newFood.asList() ]
        closestFood = min(distToFoods) if distToFoods else float('inf')
        distToGhosts = [ util.manhattanDistance(newPos,ghost.getPosition()) for ghost in newGhostStates]
        closestGhost = min(distToGhosts)  if distToGhosts else float('inf')
        """
        here we will consider three factors that might influence the new action score：
        1. the original score of the action
        2. distance to the closest food
        3. distance to the closest ghost
            a. ghost is scared: the more close the ghost is, the more score the action get
            b. ghost is not scared: the more close the ghost is, the less score the action get
        """
        # print(f"getScore:{successorGameState.getScore()}, closestFood:{1/closestFood}, signFactor:{ghostFactor}, closestGhost:{closestGhost}")
        foodScore = foodFactor * 1/max(closestFood,1)
        ghostScore = ghostFactor * (1/max(closestGhost,1))
        comprehensiveScore = successorGameState.getScore() + foodScore + ghostScore
        return comprehensiveScore


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def recursionHelper(currState, agentIndex, count):
            # 1. base case
            if currState.isWin() or currState.isLose():
                return self.evaluationFunction(currState)
            if count == 0:
                return self.evaluationFunction(currState)
            # 2. current node operation
            currentActions = currState.getLegalActions(agentIndex)
            if agentIndex == 0:         #Try to maximize the score
                bestScore = float('-inf')
                for currentAction in currentActions:
                    successorState = currState.generateSuccessor(agentIndex, currentAction)
                    tempScore = recursionHelper(successorState, agentIndex + 1, count - 1)
                    bestScore = max(bestScore, tempScore)
                return bestScore
            else:                       #Try to minimize the score
                bestScore = float('inf')
                for currentAction in currentActions:
                    successorState = currState.generateSuccessor(agentIndex, currentAction)
                    if agentIndex == agentNum - 1:
                        tempScore = recursionHelper(successorState, 0, count - 1)
                    else:
                        tempScore =  recursionHelper(successorState, agentIndex + 1, count - 1)
                    bestScore = min(bestScore, tempScore)
                return bestScore

        "*** Start excute the evaluation ***"
        # Even though the final optimal result is deterministic leaf, here we need to consider the adversarial condition
        # which means we cannot choose the optimal leaf but the relatively best we have in the worst case(adversarial)
        # print(f"agent number is:{gameState.getNumAgents()} , depth is:{self.depth}")
        agentNum = gameState.getNumAgents()
        initialActions = gameState.getLegalActions(0)
        initialCount = self.depth * agentNum
        bestScore = float('-inf')
        bestAction = None
        for initialAction in initialActions:
            initialState = gameState.generateSuccessor(0,initialAction)
            tempScore =  recursionHelper(initialState,1,initialCount-1)
            if tempScore > bestScore:
                bestAction = initialAction
                bestScore = tempScore
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(currState, agentIndex, count, alpha, beta):
            bestScore = float('-inf')
            currentActions = currState.getLegalActions(agentIndex)
            for currentAction in currentActions:
                successorState = currState.generateSuccessor(agentIndex, currentAction)
                tempScore = recursionHelper(successorState, agentIndex + 1, count - 1, alpha, beta)
                bestScore = max(bestScore, tempScore)
                if bestScore > beta:
                    return bestScore
                alpha = max(alpha, bestScore)
            return bestScore


        def minValue(currState, agentIndex, count, alpha, beta):
            bestScore = float('inf')
            currentActions = currState.getLegalActions(agentIndex)
            for currentAction in currentActions:
                successorState = currState.generateSuccessor(agentIndex, currentAction)
                if agentIndex == agentNum - 1:
                    tempScore = recursionHelper(successorState, 0, count - 1, alpha, beta)
                else:
                    tempScore =  recursionHelper(successorState, agentIndex + 1, count - 1, alpha, beta)
                bestScore = min(bestScore, tempScore)
                if bestScore < alpha:
                    return bestScore
                beta = min(beta, bestScore)

            return bestScore

        def recursionHelper(currState, agentIndex, count, alpha, beta):
            # 1. base case
            if currState.isWin() or currState.isLose():
                return self.evaluationFunction(currState)
            if count == 0:
                return self.evaluationFunction(currState)
            # 2. current node operation
            if agentIndex == 0:         #Try to maximize the score
                bestScore = maxValue(currState, agentIndex, count, alpha, beta)

            else:                       #Try to minimize the score
                bestScore = minValue(currState, agentIndex, count, alpha, beta)
            return bestScore

        "*** Start excute the evaluation ***"
        # Even though the final optimal result is deterministic leaf, here we need to consider the adversarial condition
        # which means we cannot choose the optimal leaf but the relatively best we have in the worst case(adversarial)
        # print(f"agent number is:{gameState.getNumAgents()} , depth is:{self.depth}")
        agentNum = gameState.getNumAgents()
        initialActions = gameState.getLegalActions(0)
        initialCount = self.depth * agentNum
        bestScore = float('-inf')
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        for initialAction in initialActions:
            initialState = gameState.generateSuccessor(0,initialAction)
            tempScore =  recursionHelper(initialState,1,initialCount-1,alpha,beta)
            if tempScore > bestScore:
                bestAction = initialAction
                bestScore = tempScore
            if bestScore > beta:
                return bestAction
            alpha = max(alpha, bestScore)                     # it should update the alpha value here!!!
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(currState, agentIndex, count):
            bestScore = float('-inf')
            currentActions = currState.getLegalActions(agentIndex)
            for currentAction in currentActions:
                successorState = currState.generateSuccessor(agentIndex, currentAction)
                tempScore = recursionHelper(successorState, agentIndex + 1, count - 1,)
                bestScore = max(bestScore, tempScore)
            return bestScore
        def expectValue(currState, agentIndex, count):
            expectScore = 0
            currentActions = currState.getLegalActions(agentIndex)
            for currentAction in currentActions:
                successorState = currState.generateSuccessor(agentIndex, currentAction)
                if agentIndex == agentNum - 1:
                    tempScore = recursionHelper(successorState, 0, count - 1)
                else:
                    tempScore =  recursionHelper(successorState, agentIndex + 1, count - 1)
                expectScore += tempScore
            return expectScore/max(len(currentActions),1)

        def recursionHelper(currState, agentIndex, count):
            # 1. base case
            if currState.isWin() or currState.isLose():
                return self.evaluationFunction(currState)
            if count == 0:
                return self.evaluationFunction(currState)
            # 2. current node operation
            if agentIndex == 0:         #Try to maximize the score
                bestScore = maxValue(currState, agentIndex, count)

            else:                       #Try to get the expectValue
                bestScore = expectValue(currState, agentIndex, count)
            return bestScore

        agentNum = gameState.getNumAgents()
        initialActions = gameState.getLegalActions(0)
        initialCount = self.depth * agentNum
        bestScore = float('-inf')
        bestAction = None
        for initialAction in initialActions:
            initialState = gameState.generateSuccessor(0, initialAction)
            tempScore = recursionHelper(initialState, 1, initialCount - 1)
            if tempScore > bestScore:
                bestAction = initialAction
                bestScore = tempScore
        return bestAction
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    the difference of this approach is identical to the Q1 but using currentState to evalute
    instead of using currentState + Action => successorState => evaluation
    here we will consider three factors that might influence the new action score：
        1. the original score of the action
        2. distance to the closest food
        3. distance to the closest ghost
            a. ghost is scared: the more close the ghost is, the more score the action get
            b. ghost is not scared: the more close the ghost is, the less score the action get
    """
     # Useful information you can extract from a GameState (pacman.py)
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]

    "*** YOUR CODE HERE ***"
    #the priority is to avoid the ghost then get more power and the last one will be get remaining score high
    ghostFactor = 5
    foodFactor = 3
    # print(f"newPos:{newPos} ,newFood:{newFood} ,newGhostStates:{newGhostStates} ,newScaredTimes:{newScaredTimes} !")
    if min(currScaredTimes) == 0: ghostFactor = -5
    distToFoods = [ util.manhattanDistance(currPos,food) for food in currFood.asList() ]
    closestFood = min(distToFoods) if distToFoods else float('inf')
    distToGhosts = [ util.manhattanDistance(currPos,ghost.getPosition()) for ghost in currGhostStates]
    closestGhost = min(distToGhosts)  if distToGhosts else float('inf')

    # print(f"getScore:{successorGameState.getScore()}, closestFood:{1/closestFood}, signFactor:{ghostFactor}, closestGhost:{closestGhost}")
    foodScore = foodFactor * 1/max(closestFood,1)
    ghostScore = ghostFactor * (1/max(closestGhost,1))
    comprehensiveScore = currentGameState.getScore() + foodScore + ghostScore
    return comprehensiveScore

# Abbreviation
better = betterEvaluationFunction
