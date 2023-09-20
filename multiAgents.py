from multiagent.util import manhattanDistance
from multiagent.game import Directions
import random
import multiagent.util as util
import copy

from multiagent.game import Agent


class ReflexAgent(Agent):#이건 평가하는 기능이고
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):#평가 함수에 따라 최적의 옵션 선택 gamestate를 실행하고 directions 반환
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        
        # Choose one of the best actions
        scores =better# [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):#평가함수 원하면 갈아끼셈
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
        score = 0

        closestGhostPosition = newGhostStates[0].configuration.pos
        closestGhost = manhattanDistance(newPos, closestGhostPosition)

        # Minimize distance from pacman to food
        newFoodPositions = newFood.asList()
        foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodPositions]

        if len(foodDistances) == 0:
            return 0

        closestFood = min(foodDistances)

        # Stop action would reduce score because of the pacman's timer constraint
        if action == 'Stop':
            score -= 50

        return successorGameState.getScore() + closestGhost / (closestFood * 10) + score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='3'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.constant_depth = int(depth)
        self.level = 0


class MinimaxAgent(MultiAgentSearchAgent):#미니맥스 트리
    """
      We have implemented MinimaxAgent for you. Read it carefully.
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):  agent의 동작이 합리적인지 판단해준다. index가 0이면 팩맨이고 1보다 크면 유령이다.
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # Format of result = [score, action]
        result = self.get_value(gameState, 0, 0)

        # Return the action from result
        return result[1]

    def get_value(self, gameState, index, depth):
        """
        Returns value as pair of [score, action] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # Terminal states:
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:#현재 깊이가 설정한 depth와 같거나 이기거나 진 상태를 확인
            return gameState.getScore(), ""#getLegal = isWin or isLose

        # Max-agent: Pacman has index = 0
        if index == 0:#팩맨 차례면 max value 리턴
            return self.max_value(gameState, index, depth)

        # Min-agent: Ghost has index > 0
        else:#고스트 차례면 min_vlaue 리턴
            return self.min_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):#amx value 분석
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = gameState.getLegalActions(index)#legal action을 취하는지 확인 현재 상태에서 가능한 모든 움직임의 반복문
        max_value = float("-inf")#value랑 액션은 여기 나와있네 현재 상태에서 가능한 움직임들 저장
        max_action = ""

        for action in legalMoves:#현재 상태에서 파생될 수 있는 상태들에 대한 min_val 중 최대값
            successor = gameState.generateSuccessor(index, action)#다음단계
            successor_index = index + 1 #index
            successor_depth = depth   #depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0] #현재 value 가져오고

            if current_value > max_value: #만약에 현재값이 기존 maxvalue보다 크면 현재 value가 maxvalue 가 되고 action은 maxaction으로
                max_value = current_value
                max_action = action

        return max_value, max_action

    def min_value(self, gameState, index, depth):
        """
        Returns the min utility value-action for min-agent
        """
        legalMoves = gameState.getLegalActions(index) #현재 유령이 취할 수 있는 움직임
        min_value = float("inf")
        min_action = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action) #현재 유령이 취할 수 있는 움직임 
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0]

            if current_value < min_value:
                min_value = current_value
                min_action = action

        return min_value, min_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
                # Format of result = [score, action]
        result = self.get_value(gameState, 0, 0,float("inf"),float("-inf"))

        # Return the action from result
        return result[1]  
    def get_value(self, gameState, index, depth,alpha,beta): # 이 부분은 알파 베타만 추가하고 바꿀게 없는거 같음
            """
            Returns value as pair of [score, action] based on the different cases:
            1. Terminal state
            2. Max-agent
            3. Min-agent
            """
            # Terminal states:
            if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:#현재 깊이가 설정한 depth와 같거나 이기거나 진 상태를 확인
                return gameState.getScore(), ""#getLegal = isWin or isLose

            # Max-agent: Pacman has index = 0
            if index == 0:#팩맨 차례면 max value 리턴
                return self.max_value(gameState, index, depth,alpha,beta)

            # Min-agent: Ghost has index > 0
            else:#고스트 차례면 min_vlaue 리턴
                return self.min_value(gameState, index, depth,alpha,beta)

    def max_value(self, gameState, index, depth,alpha,beta):#amx value 분석
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = gameState.getLegalActions(index)#legal action을 취하는지 확인 현재 상태에서 가능한 모든 움직임의 반복문
        max_value = float("-inf")#value랑 액션은 여기 나와있네 현재 상태에서 가능한 움직임들 저장
        alpha = float("-inf")
        beta =  float("inf")
        max_action = ""

        for action in legalMoves:#현재 상태에서 파생될 수 있는 상태들에 대한 min_val 중 최대값
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1 #index
            successor_depth = depth   #depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth, alpha, beta)[0] #현재 value 가져오고

            if current_value > max_value: #best action의 알고리즘
                max_value = current_value
                max_action = action
            #prunning
            if max_value > beta:
                return max_value, max_action
            else:
                alpha=max_value

        return max_value, max_action #각 부분 실행마다 max value랑 max action을 리턴해준다.

    def min_value(self, gameState, index, depth, alpha,beta):
        """
        Returns the min utility value-action for min-agent
        """
        legalMoves = gameState.getLegalActions(index) #현재 유령이 취할 수 있는 움직임
        min_value = float("inf")
        alpha =float("-inf")
        beta = float("inf")
        min_action = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action) #현재 유령이 취할 수 있는 움직임 
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth,alpha,beta)[0]

            if current_value < min_value:
                min_value = current_value
                min_action = action
            if min_value < alpha:
                return min_value, min_action
            else:
                beta=min_value
        return min_value, min_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):  agent의 동작이 합리적인지 판단해준다. index가 0이면 팩맨이고 1보다 크면 유령이다.
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
    
        # Format of result = [score, action]
        result = self.get_value(gameState, 0, 0)

        # Return the action from result
        return result[1]

    def get_value(self, gameState, index, depth):
        """
        Returns value as pair of [score, action] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # Terminal states:
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:#현재 깊이가 설정한 depth와 같거나 이기거나 진 상태를 확인
            return gameState.getScore(), ""#getLegal = isWin or isLose

        # Max-agent: Pacman has index = 0
        if index == 0:#팩맨 차례면 max value 리턴
            return self.max_value(gameState, index, depth)

        # Min-agent: Ghost has index > 0
        else:#고스트 차례면 min_vlaue 리턴
            return self.expectimax_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):#amx value 분석
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = gameState.getLegalActions(index)#legal action을 취하는지 확인 현재 상태에서 가능한 모든 움직임의 반복문
        max_value = float("-inf")#value랑 액션은 여기 나와있네 현재 상태에서 가능한 움직임들 저장
        max_action = ""

        for action in legalMoves:#현재 상태에서 파생될 수 있는 상태들에 대한 min_val 중 최대값
            successor = gameState.generateSuccessor(index, action)#다음단계
            successor_index = index + 1 #index
            successor_depth = depth   #depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0] #현재 value 가져오고

            if current_value > max_value: #만약에 현재값이 기존 maxvalue보다 크면 현재 value가 maxvalue 가 되고 action은 maxaction으로
                max_value = current_value
                max_action = action

        return max_value, max_action

    def expectimax_value(self, gameState, index, depth):
        """
        Returns the min utility value-action for min-agent
        """
        legalMoves = gameState.getLegalActions(index) #현재 유령이 취할 수 있는 움직임
        expect_value = 0
        expect_action = ""
        successor_probability = float(1.0/len(legalMoves)) #어느쪽으로 갈지 left, right up, down stop 경우의 수에 따라 나눠줘서 큰 값을 찾으니까
        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action) #현재 유령이 취할 수 있는 움직임 
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0]
            
            expect_value += successor_probability*current_value

        return expect_value, expect_action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


#  def evaluationFunction(self, currentGameState, action):#평가함수 원하면 갈아끼셈
#      
#         # Useful information you can extract from a GameState (pacman.py)
#         successorGameState = currentGameState.generatePacmanSuccessor(action)
#         newPos = successorGameState.getPacmanPosition()
#         newFood = successorGameState.getFood()
#         newGhostStates = successorGameState.getGhostStates()
#         newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

#         "*** YOUR CODE HERE ***"
#         score = 0

#         closestGhostPosition = newGhostStates[0].configuration.pos
#         closestGhost = manhattanDistance(newPos, closestGhostPosition)

#         # Minimize distance from pacman to food
#         newFoodPositions = newFood.asList()
#         foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodPositions]

#         if len(foodDistances) == 0:
#             return 0

#         closestFood = min(foodDistances)

#         # Stop action would reduce score because of the pacman's timer constraint
#         if action == 'Stop':
#             score -= 50

#         return successorGameState.getScore() + closestGhost / (closestFood * 10) + score

# Abbreviation
better = betterEvaluationFunction
