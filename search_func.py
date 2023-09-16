from lib2to3.pytree import Node
from queue import Empty
import queue
from pacman.game import Directions
from pacman.util import raiseNotDefined
import util
from heuristics import nullHeuristic
from heuristics import manhattanHeuristic
import external_lib


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Are we reaching a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    You can also refer to the function "tinyMazeSearch"
    """
    "*** YOUR CODE HERE ***"
    #dfs should make list and tree, every list save state(x,y) 
    #fringe means collection of nodes want expand
    #maybe we need check path cost , let's try
    #searchAgent에 함수 있음
    startPoint=problem.getStartState()
    fringe=util.Stack() #확장 가능한 노드들 다 저장
    fringe.push((startPoint,[]))#시작점 넣고 체크 시작
    visited=[] #방문한 노드 체크하는 리스트
    while fringe is not Empty:
        node,direction=fringe.pop() #if goal is (1,1) node is (5,5) go on, get every succesors
        visited.append(node) #방문체크 하고 다음 자식노드들 체크 해야지 succesor
        if(problem.isGoalState(node)):#만약에 갱신된 node가 goal이라면 그때부터 스택에서 이동명령을 내리면 된다.
            print(direction)
            return direction
        for succsessor ,Action,cost in problem.getSuccessors((node)):
            if not succsessor in visited:
                fringe.push((succsessor,direction+[Action]))
                #print(direction)
    print("goal is not exist")#결과값에 접근이 실패했다면
    return []

        
#    raiseNotDefined()  # DONT FORGET TO COMMENT THIS LINE AFTER YOU IMPLEMENT THIS FUNCTION!!!!!!

                 #a
            #b          c
    #d                          e
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    startPoint=problem.getStartState()
    fringe=util.Queue() #확장 가능한 노드들 다 저장
    fringe.push((startPoint,[]))#시작점 넣고 체크 시작
    visited=[] #방문한 노드 체크하는 리스트
    while fringe is not Empty:
        node,direction=fringe.pop() #if goal is (1,1) node is (5,5) go on, get every succesors
        visited.append(node) #방문체크 하고 다음 자식노드들 체크 해야지 succesor
        if(problem.isGoalState(node)):#만약에 갱신된 node가 goal이라면 그때부터 스택에서 이동명령을 내리면 된다.
            print(direction)
            return direction
        for succsessor ,Action,cost in problem.getSuccessors((node)):
            if not succsessor in visited:
                fringe.push((succsessor,direction+[Action]))
                #print(direction)
    print("goal is not exist")#결과값에 접근이 실패했다면
    return []
    #raiseNotDefined()  # DONT FORGET TO COMMENT THIS LINE AFTER YOU IMPLEMENT THIS FUNCTION!!!!!!


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    len=0
    startPoint=problem.getStartState()
    fringe=util.PriorityQueue() #확장 가능한 노드들 다 저장
    fringe.push((startPoint,[]),1)#priorty 큐니까 cost도 넣어주는데 여기선 다 1 아닐까?
    visited=[] #방문한 노드 체크하는 리스트
    while fringe is not Empty:
        node,direction=fringe.pop() #if goal is (1,1) node is (5,5) go on, get every succesors
        len-=1
        visited.append(node) #방문체크 하고 다음 자식노드들 체크 해야지 succesor
        if(problem.isGoalState(node)):#만약에 갱신된 node가 goal이라면 그때부터 스택에서 이동명령을 내리면 된다.
            print(direction)
            print("from start to goal:",len)
            return direction
        for succsessor ,Action,cost in problem.getSuccessors((node)):
            if not succsessor in visited:
                fringe.push((succsessor,direction+[Action]),cost)
                print("cost is:",cost)
                len+=cost
                #print(direction)
    print("goal is not exist")#결과값에 접근이 실패했다면
    return []
    #raiseNotDefined()  # DONT FORGET TO COMMENT THIS LINE AFTER YOU IMPLEMENT THIS FUNCTION!!!!!!


def aStarSearch(problem, heuristic=manhattanHeuristic):#problem은 position , action, cost로 이루어져 있다.
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #think, A star should have 1.heuritsic value(ex expect value of current pos to goal) 2.node cost from start point! 
    #so maybe I need to get goal point at first, and tracking it
    #if node is at target position, mean find, if openList len is 0, fail to find
    #check g,h val in fringe, if no states exist for expand, finsh func
    startPoint=problem.getStartState()
    fringe=util.PriorityQueue() #
    fringe.push((startPoint,[],0),heuristic(startPoint,problem))#fringe에는 좌표,방향,cost 값이 필요할듯
    print("휴리스틱 값:",heuristic(startPoint,problem))
    visited=dict() #key value값으로 좌표마다 값을 가진 일종의 graph 형태가 되야한다.
    #raiseNotDefined()  # DONT FORGET TO COMMENT THIS LINE AFTER YOU IMPLEMENT THIS FUNCTION!!!!!!
    while not fringe is Empty:
        node,direction,cost=fringe.pop()
        visited[node]=cost #key value  
        print("휴리스틱 값:",heuristic(node,problem))

        if problem.isGoalState(node):
            return direction
        # 0<=h(n)<h*(n)
        for succsessor ,Action,curr_cost in problem.getSuccessors((node)):
            cost_from_start=cost+curr_cost
            total_cost=cost_from_start+heuristic(node,problem)#F=G+H
            #방문한 적이 없거나 갱신한 F,G,H,Parent 노드 값이 현재 갱신하는 값보다 작을경우 
            if (not succsessor in visited) or (succsessor in visited and visited[succsessor]>total_cost):
                visited[succsessor]=cost_from_start
                fringe.push((succsessor,direction+[Action],cost_from_start),total_cost)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
