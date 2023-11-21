# Fundamental-of-Artificial-Intelligence
## Implement Depth First Search Traversal of a Graph 01
### Program 
```py
#import defaultdict
from collections import defaultdict
def dfs(graph,start,visited,path):
 path.append(start)
 visited[start]=True
 for neighbour in graph[start]:
 if visited[neighbour]==False:
 dfs(graph,neighbour,visited,path)
 visited[neighbour]=True
 return path
graph=defaultdict(list)
n,e=map(int,input().split())
for i in range(e):
 u,v=map(str,input().split())
 graph[u].append(v)
 graph[v].append(u)
#print(graph)
start='0' # The starting node is 0
visited=defaultdict(bool)
path=[]
traversedpath=dfs(graph,start,visited,path)
print(traversedpath) 
```
## Implement Breadth First Search Traversal of a Graph 02
```py
from collections import deque
from collections import defaultdict
def bfs(graph,start,visited,path):
 queue = deque()
 path.append(start)
 queue.append(start)
 visited[start] = True
 while len(queue) != 0:
 tmpnode = queue.popleft()
 for neighbour in graph[tmpnode]:
 if visited[neighbour] == False:
 path.append(neighbour)
 queue.append(neighbour)
 visited[neighbour] = True
 return path
graph = defaultdict(list)
v,e = map(int,input().split())
for i in range(e):
 u,v = map(str,input().split())
 graph[u].append(v)
 graph[v].append(u)
start = '0'
path = []
visited = defaultdict(bool)
traversedpath = bfs(graph,start,visited,path)
print(traversedpath)
```
## Implement A* search algorithm for a Graph 03
```py
from collections import defaultdict
H_dist ={}
def aStarAlgo(start_node, stop_node):
 open_set = set(start_node)
 closed_set = set()
 g = {} #store distance from starting node
 parents = {} # parents contains an adjacency map of all nodes
 #distance of starting node from itself is zero
 g[start_node] = 0
 #start_node is root node i.e it has no parent nodes
 #so start_node is set to its own parent node
 parents[start_node] = start_node
 while len(open_set) > 0:
 n = None
 #node with lowest f() is found
 for v in open_set:
15
 if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
 n = v
 if n == stop_node or Graph_nodes[n] == None:
 pass
 else:
 for (m, weight) in get_neighbors(n):
 #nodes 'm' not in first and last set are added to first
 #n is set its parent
 if m not in open_set and m not in closed_set:
 open_set.add(m)
 parents[m] = n
 g[m] = g[n] + weight
 #for each node m,compare its distance from start i.e g(m) to the
 #from start through n node
 else:
 if g[m] > g[n] + weight:
 #update g(m)
 g[m] = g[n] + weight
 #change parent of m to n
 parents[m] = n
 #if m in closed set,remove and add to open
 if m in closed_set:
 closed_set.remove(m)
 open_set.add(m)
 if n == None:
 print('Path does not exist!')
 return None

 # if the current node is the stop_node
 # then we begin reconstructin the path from it to the start_node
 if n == stop_node:
 path = []
 while parents[n] != n:
 path.append(n)
 n = parents[n]
 path.append(start_node)
 path.reverse()
 print('Path found: {}'.format(path))
 return path
 # remove n from the open_list, and add it to closed_list
 # because all of his neighbors were inspected
 open_set.remove(n)
 closed_set.add(n)
 print('Path does not exist!')
 return None
#define fuction to return neighbor and its distance
16
#from the passed node
def get_neighbors(v):
 if v in Graph_nodes:
 return Graph_nodes[v]
 else:
 return None
def heuristic(n):
 return H_dist[n]
graph = defaultdict(list)
n,e = map(int,input().split())
for i in range(e):
 u,v,cost = map(str,input().split())
 t=(v,float(cost))
 graph[u].append(t)
 t1=(u,float(cost))
 graph[v].append(t1)
for i in range(n):
 node,h=map(str,input().split())
 H_dist[node]=float(h)
print(H_dist)
Graph_nodes=graph
print(graph)
aStarAlgo('A', 'J')
```
## Implement Simple Hill Climbing Algorithm 04
```py
import random
import string
def generate_random_solution(answer):
 l=len(answer)
 return [random.choice(string.printable) for _ in range(l)]
def evaluate(solution,answer):
 print(solution)
 target=list(answer)
 diff=0
 for i in range(len(target)):
 s=solution[i]
 t=target[i]
 diff +=abs(ord(s)-ord(t))
 return diff
def mutate_solution(solution):
 ind=random.randint(0,len(solution)-1)
 solution[ind]=random.choice(string.printable)
 return solution
def SimpleHillClimbing():
 answer="Artificial Intelligence"
 best=generate_random_solution(answer)
 best_score=evaluate(best,answer)
 while True:
 print("Score:",best_score," Solution : ","".join(best))
 if best_score==0:
 break
 new_solution=mutate_solution(list(best))
 score=evaluate(new_solution,answer)
 if score<best_score:
 best=new_solution
 best_score=score
#answer="Artificial Intelligence"
#print(generate_random_solution(answer))
#solution=generate_random_solution(answer)
#print(evaluate(solution,answer))
SimpleHillClimbing()
```
## Implement Minimax Search Algorithm for a Simple TIC-TAC-TOE game 05
```py
import time
class Game:
 def __init__(self):
 self.initialize_game()
 def initialize_game(self):
 self.current_state = [['.','.','.'],
 ['.','.','.'],
27
 ['.','.','.']]
 # Player X always plays first
 self.player_turn = 'X'
 def draw_board(self):
 for i in range(0, 3):
 for j in range(0, 3):
 print('{}|'.format(self.current_state[i][j]), end=" ")
 print()
 print()
 def is_valid(self, px, py):
 if px < 0 or px > 2 or py < 0 or py > 2:
 return False
 elif self.current_state[px][py] != '.':
 return False
 else:
 return True
 def is_end(self):
 # Vertical win
 for i in range(0, 3):
 if (self.current_state[0][i] != '.' and
 self.current_state[0][i] == self.current_state[1][i] and
 self.current_state[1][i] == self.current_state[2][i]):
 return self.current_state[0][i]
 # Horizontal win
 for i in range(0, 3):
 if (self.current_state[i] == ['X', 'X', 'X']):
 return 'X'
 elif (self.current_state[i] == ['O', 'O', 'O']):
 return 'O'
 # Main diagonal win
 if (self.current_state[0][0] != '.' and
 self.current_state[0][0] == self.current_state[1][1] and
 self.current_state[0][0] == self.current_state[2][2]):
 return self.current_state[0][0]
 # Second diagonal win
 if (self.current_state[0][2] != '.' and
 self.current_state[0][2] == self.current_state[1][1] and
 self.current_state[0][2] == self.current_state[2][0]):
 return self.current_state[0][2]
 # Is the whole board full?
 for i in range(0, 3):
 for j in range(0, 3):
 # There's an empty field, we continue the game
 if (self.current_state[i][j] == '.'):
 return None
 # It's a tie!
28
 return '.'
 def max(self):
 # Possible values for maxv are:
 # -1 - loss
 # 0 - a tie
 # 1 - win
 # We're initially setting it to -2 as worse than the worst case:
 maxv = -2
 px = None
 py = None
 result = self.is_end()
 # If the game came to an end, the function needs to return
 # the evaluation function of the end. That can be:
 # -1 - loss
 # 0 - a tie
 # 1 - win
 if result == 'X':
 return (-1, 0, 0)
 elif result == 'O':
 return (1, 0, 0)
 elif result == '.':
 return (0, 0, 0)
 for i in range(0, 3):
 for j in range(0, 3):
 if self.current_state[i][j] == '.':
 # On the empty field player 'O' makes a move and calls Min
 # That's one branch of the game tree.
 self.current_state[i][j] = 'O'
 (m, min_i, min_j) = self.min()
 # Fixing the maxv value if needed
 if m > maxv:
 maxv = m
 px = i
 py = j
 # Setting back the field to empty
 self.current_state[i][j] = '.'
 return (maxv, px, py)
 def min(self):
 # Possible values for minv are:
 # -1 - win
 # 0 - a tie
 # 1 - loss
 # We're initially setting it to 2 as worse than the worst case:
 minv = 2
 qx = None
 qy = None
29
 result = self.is_end()
 if result == 'X':
 return (-1, 0, 0)
 elif result == 'O':
 return (1, 0, 0)
 elif result == '.':
 return (0, 0, 0)
 for i in range(0, 3):
 for j in range(0, 3):
 if self.current_state[i][j] == '.':
 self.current_state[i][j] = 'X'
 (m, max_i, max_j) = self.max()
 if m < minv:
 minv = m
 qx = i
 qy = j
 self.current_state[i][j] = '.'
 return (minv, qx, qy)
 def play(self):
 while True:
 self.draw_board()
 self.result = self.is_end()
 # Printing the appropriate message if the game has ended
 if self.result != None:
 if self.result == 'X':
 print('The winner is X!')
 elif self.result == 'O':
 print('The winner is O!')
 elif self.result == '.':
 print("It's a tie!")
 self.initialize_game()
 return
 # If it's player's turn
 if self.player_turn == 'X':
 while True:
 start = time.time()
 (m, qx, qy) = self.min()
 end = time.time()
 print('Evaluation time: {}s'.format(round(end - start, 7)))
 print('Recommended move: X = {}, Y = {}'.format(qx, qy))
 px = int(input('Insert the X coordinate: '))
 py = int(input('Insert the Y coordinate: '))
 (qx, qy) = (px, py)
 if self.is_valid(px, py):
 self.current_state[px][py] = 'X'
 self.player_turn = 'O'
30
 break
 else:
 print('The move is not valid! Try again.')
 # If it's AI's turn
 else:
 (m, px, py) = self.max()
 self.current_state[px][py] = 'O'
 self.player_turn = 'X'
def main():
 g = Game()
 g.play()
if __name__ == "__main__":
 main()
```
## Implement Alpha-beta pruning of Minimax Search Algorithm for a
Simple TIC-TAC-TOE game 06
```py
import time
class Game:
 def __init__(self):
 self.initialize_game()
 def initialize_game(self):
 self.current_state = [['.','.','.'],
 ['.','.','.'],
 ['.','.','.']]
33
 # Player X always plays first
 self.player_turn = 'X'
 def draw_board(self):
 for i in range(0, 3):
 for j in range(0, 3):
 print('{}|'.format(self.current_state[i][j]), end=" ")
 print()
 print()
 def is_valid(self, px, py):
 if px < 0 or px > 2 or py < 0 or py > 2:
 return False
 elif self.current_state[px][py] != '.':
 return False
 else:
 return True
 def is_end(self):
 # Vertical win
 for i in range(0, 3):
 if (self.current_state[0][i] != '.' and
 self.current_state[0][i] == self.current_state[1][i] and
 self.current_state[1][i] == self.current_state[2][i]):
 return self.current_state[0][i]
 # Horizontal win
 for i in range(0, 3):
 if (self.current_state[i] == ['X', 'X', 'X']):
 return 'X'
 elif (self.current_state[i] == ['O', 'O', 'O']):
 return 'O'
 # Main diagonal win
 if (self.current_state[0][0] != '.' and
 self.current_state[0][0] == self.current_state[1][1] and
 self.current_state[0][0] == self.current_state[2][2]):
 return self.current_state[0][0]
 # Second diagonal win
 if (self.current_state[0][2] != '.' and
 self.current_state[0][2] == self.current_state[1][1] and
 self.current_state[0][2] == self.current_state[2][0]):
 return self.current_state[0][2]
 # Is the whole board full?
 for i in range(0, 3):
 for j in range(0, 3):
 # There's an empty field, we continue the game
 if (self.current_state[i][j] == '.'):
 return None
 # It's a tie!
34
 return '.'
 def max_alpha_beta(self, alpha, beta):
 maxv = -2
 px = None
 py = None
 result = self.is_end()
 if result == 'X':
 return (-1, 0, 0)
 elif result == 'O':
 return (1, 0, 0)
 elif result == '.':
 return (0, 0, 0)
 for i in range(0, 3):
 for j in range(0, 3):
 if self.current_state[i][j] == '.':
 self.current_state[i][j] = 'O'
 (m, min_i, in_j) = self.min_alpha_beta(alpha, beta)
 if m > maxv:
 maxv = m
 px = i
 py = j
 self.current_state[i][j] = '.'
 # Next two ifs in Max and Min are the only difference between regular algorithm
and minimax
 if maxv >= beta:
 return (maxv, px, py)
 if maxv > alpha:
 alpha = maxv
 return (maxv, px, py)
 def min_alpha_beta(self, alpha, beta):
 minv = 2
 qx = None
 qy = None
 result = self.is_end()
 if result == 'X':
 return (-1, 0, 0)
 elif result == 'O':
 return (1, 0, 0)
 elif result == '.':
 return (0, 0, 0)
 for i in range(0, 3):
 for j in range(0, 3):
 if self.current_state[i][j] == '.':
 self.current_state[i][j] = 'X'
 (m, max_i, max_j) = self.max_alpha_beta(alpha, beta)
35
 if m < minv:
 minv = m
 qx = i
 qy = j
 self.current_state[i][j] = '.'
 if minv <= alpha:
 return (minv, qx, qy)
 if minv < beta:
 beta = minv
 return (minv, qx, qy)
 def play_alpha_beta(self):
 while True:
 self.draw_board()
 self.result = self.is_end()
 if self.result != None:
 if self.result == 'X':
 print('The winner is X!')
 elif self.result == 'O':
 print('The winner is O!')
 elif self.result == '.':
 print("It's a tie!")
 self.initialize_game()
 return
 if self.player_turn == 'X':
 while True:
 start = time.time()
 (m, qx, qy) = self.min_alpha_beta(-2, 2)
 end = time.time()
 print('Evaluation time: {}s'.format(round(end - start, 7)))
 print('Recommended move: X = {}, Y = {}'.format(qx, qy))
 px = int(input('Insert the X coordinate: '))
 py = int(input('Insert the Y coordinate: '))
 qx = px
 qy = py
 if self.is_valid(px, py):
 self.current_state[px][py] = 'X'
 self.player_turn = 'O'
 break
 else:
 print('The move is not valid! Try again.')
 else:
 (m, px, py) = self.max_alpha_beta(-2, 2)
 self.current_state[px][py] = 'O'
36
 self.player_turn = 'X'
def main():
 g = Game()
 g.play_alpha_beta()
if __name__ == "__main__":
 main() 
```
## Solve Cryptarithmetic Problem,a CSP(Constraint Satisfaction
Problem) using Python 07
```py
from itertools import permutations
def solve_cryptarithmetic():
 for perm in permutations(range(10), 8):
 S, E, N, D, M, O, R, Y = perm
 # Check for leading zeros
 if S == 0 or M == 0:
 continue
 # Check the equation constraints
 SEND = 1000 * S + 100 * E + 10 * N + D
 MORE = 1000 * M + 100 * O + 10 * R + E
 MONEY = 10000 * M + 1000 * O + 100 * N + 10 * E + Y
 if SEND + MORE == MONEY:
 return SEND, MORE, MONEY
 return None
solution = solve_cryptarithmetic()
if solution:
 SEND, MORE, MONEY = solution
 print(f'SEND = {SEND}')
 print(f'MORE = {MORE}')
 print(f'MONEY = {MONEY}')
else:
 print("No solution found.")
```
## Solve Wumpus World Problem using Python demonstrating Inferences
from Propositional Logic 08
```py
wumpus=[["Save","Breeze","PIT","Breeze"],
 ["Smell","Save","Breeze","Save"],
 ["WUMPUS","GOLD","PIT","Breeze"],
 ["Smell","Save","Breeze","PIT"]]
row=0
column=0
arrow=True
player=True
score=0
while(player):
 choice=input("press u to move up\npress d to move down\npress l to move left\npress r to
move right\n")
 if choice == "u":
 if row != 0:
 row-=1
 else:
 print("move denied")
 print("current location: ",wumpus[row][column],"\n")
 elif choice == "d" :
 if row!=3:
 row+=1
 else:
 print("move denied")
 print("current location: ",wumpus[row][column],"\n")
 elif choice == "l" :
 if column!=0:
 column-=1
 else:
 print("move denied")
 print("current location: ",wumpus[row][column],"\n")
 elif choice == "r" :
 if column!=3:
 column+=1
 else:
 print("move denied")
 print("current location: ",wumpus[row][column],"\n")
 else:
 print("move denied")
 if wumpus[row][column]=="Smell" and arrow != False:
 arrow_choice=input("do you want to throw an arrow-->\npress y to throw\npress n to
save your arrow\n")
 if arrow_choice == "y":
 arrow_throw=input("press u to throw up\npress d to throw down\npress l to throw
left\npress r to throw right\n")
 if arrow_throw == "u":
42
 if wumpus[row-1][column] == "WUMPUS":
 print("wumpus killed!")
 score+=1000
 print("score: ",score)
 wumpus[row-1][column] = "Save"
 wumpus[1][0]="Save"
 wumpus[3][0]="Save"
 else:
 print("arrow wasted...")
 score-=10
 print("score: ",score)
 elif arrow_throw == "d":
 if wumpus[row+1][column] == "WUMPUS":
 print("wumpus killed!")
 score+=1000
 print("score: ",score)
 wumpus[row+1][column] = "Save"
 wumpus[1][0]="Save"
 wumpus[3][0]="Save"
 else:
 print("arrow wasted...")
 score-=10
 print("score: ",score)
 elif arrow_throw == "l":
 if wumpus[row][column-1] == "WUMPUS":
 print("wumpus killed!")
 score+=1000
 print("score: ",score)
 wumpus[row][column-1] = "Save"
 wumpus[1][0]="Save"
 wumpus[3][0]="Save"
 else:
 print("arrow wasted...")
 score-=10
 print("score: ",score)
 elif arrow_throw == "r":
 if wumpus[row][column+1] == "WUMPUS":
 print("wumpus killed!")
 score+=1000
 print("score: ",score)
 wumpus[row][column+1] = "Save"
 wumpus[1][0]="Save"
 wumpus[3][0]="Save"
 else:
 print("arrow wasted...")
 score-=10
43
 print("score: ",score)
 arrow=False
 if wumpus[row][column] == "WUMPUS" :
 score-=1000
 print("\nWumpus here!!\n You Die\nAnd your score is: ",score
 ,"\n")
 break
 if(wumpus[row][column]=='GOLD'):
 score+=1000
 print("GOLD FOUND!You won....\nYour score is: ",score,"\n")
 break
 if(wumpus[row][column]=='PIT'):
 score-=1000
 print("Ahhhhh!!!!\nYou fell in pit.\nAnd your score is: ",score,"\n")
 break 
```
