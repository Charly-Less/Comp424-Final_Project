"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view,LINES)
from constants import *
import numpy as np
import time

#WIN_SCORE = 10**6
MAX_DEPTH = 2 ## 3 is too slow
@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"

    def compute_line_potential(self,line, owners,player,opponent):
        """ Computes the line potential score for one line"""


        my_count = 0
        opp_count = 0
        multiplier = [0,5,20,100]
        for r,c in line:
            if owners[r,c] == OBSTACLE:
                return 0
            elif owners[r,c] == player:
                my_count += 1
                
            elif owners[r,c] == opponent:
                opp_count += 1

        if opp_count != 0 and my_count != 0:
            return 0
        elif opp_count == 0:
            return multiplier[my_count]
        else:
            return -multiplier[opp_count]
     

    def line_potential(self,board,player,opponent):
        """Sums the line_potential for all lines"""
        owners, sizes = top_view(board)
        score = 0
        for line in LINES[WIN_LENGTH]:
            score += self.compute_line_potential(line,owners,player,opponent)
        return score  
    
    def largest_in_reserve(self, reserves, who):
        """Largest piece size 'who' can still place"""
        for s in (LARGE, MEDIUM, SMALL):
            if reserves[who, s] > 0:
                return s
        return -1
    
    def count_threats(self, owners, sizes, reserves, who, other):
        """Lines where 'who' has 3 and can complete the 4th next turn."""
        n = 0 # number of threats
        biggest = self.largest_in_reserve(reserves, who)
        for line in LINES[WIN_LENGTH]: # List of every possible winning line on the board
            cells = [owners[r,c] for r,c in line] # owners = who owns top piece on each square
            if OBSTACLE in cells or cells.count(who) != 3: # OBSTACLE = blocked square
                continue
            r,c = next((r,c) for r,c in line if owners[r,c] != who)
            if owners[r,c] == EMPTY:
                n += 1
            elif owners[r,c] == other and sizes[r,c] < biggest: # opponents piece is on top of 4th square and their piece is smaller than the largest pieces who has in reserve 
                n += 1
        return n
    
    def threat_score(self, owners, sizes, reserves, player, opponent):
        mine = self.count_threats(owners, sizes, reserves, player, opponent)
        theirs = self.count_threats(owners, sizes, reserves, opponent, player)
        score = 50 * mine - 60 * theirs
        if mine >= 2:
            score += 200
        if theirs >= 2:
            score -=250
        return score
    

    def uncover_risk(self, board, owners, sizes, player, opponent):
        """Penalty for our top pieces that are stuck covering opponent where they can win."""
        penalty = 0
        for line in LINES[WIN_LENGTH]: # Go through all 10 possible winning lines
            cells = [owners[r,c] for r,c in line] # Make a list of who owns the top of each of the line's 4 squares
            if OBSTACLE in cells or cells.count(opponent) != 3: # Skip if they don't have 3 of 4 squares, or if the obstacle is there
                continue
            r,c = next((r,c) for r,c in line if owners[r,c] != opponent) # Find 4th square and store its row and column 
            if owners[r,c] != player: # If 4th square isn't yours on top, skip
                continue
            below = [board[r,c,s] for s in range(sizes[r,c]) if board[r,c,s] != EMPTY] # List the pieces under your top piece - if large covers Brown's small, below = [2]
            if below and below[-1] == opponent: # Checks if something underneath at all, below[-1] is the biggest piece under yours, if opponents lifting would complete their line
                penalty -= 30 
        return penalty



    
    def evaluate(self,board,reserves,player,opponent):
        """Overall evaluation function."""
        owners, sizes = top_view(board)
        return self.line_potential(board,player,opponent) + self.threat_score(owners, sizes, reserves, player, opponent) + self.uncover_risk(board, owners, sizes, player, opponent)

    def minimax(self,board,reserves, depth, alpha, beta, current_player, root_player):
        """Minimax with alpha-beta pruning"""
        last_player = 3 - current_player
        winner = check_winner(board, last_player , WIN_LENGTH)

        if winner == root_player:
            return 1000000 + depth

        elif winner == 3 - root_player:
            return -1000000 - depth

        elif depth == 0:
            return self.evaluate(board,reserves, root_player, 3 - root_player)

        is_max_level = (current_player == root_player)
        if is_max_level:

            value = float('-inf')

            for move in get_valid_moves(board,reserves,current_player,ALLOW_MOVES):

                new_board, new_reserves = copy_state(board, reserves)
                execute_move(new_board,new_reserves,current_player, move,ALLOW_MOVES)

                
                score = self.minimax(
                    new_board,
                    new_reserves,
                    depth - 1,
                    alpha,
                    beta,
                    3 - current_player,
                    root_player
                )

                value = max(value, score)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break      # beta cutoff

            return value

        else:

            value = float('inf')

            for move in get_valid_moves(board,reserves,current_player,ALLOW_MOVES):
            
                new_board, new_reserves = copy_state(board, reserves)
                execute_move(new_board,new_reserves,current_player, move,ALLOW_MOVES)
                
                score = self.minimax(
                    new_board,
                    new_reserves,
                    depth - 1,
                    alpha,
                    beta,
                    3 - current_player,
                    root_player
                )

                value = min(value, score)
                beta = min(beta, value)

                if beta <= alpha:
                    break      # alpha cutoff

            return value
            


    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """
        start = time.perf_counter()
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.
        
        best_moves = []
        best_score = float('-inf')
        for move in get_valid_moves(board,reserves,player,ALLOW_MOVES):
            new_board, new_reserves = copy_state(board, reserves)

            execute_move(
                new_board,
                new_reserves,
                player,
                move,
                ALLOW_MOVES
            )
            score = self.minimax(
            new_board,
            new_reserves,
            MAX_DEPTH - 1,
            float("-inf"),
            float("inf"),
            3 - player,
            player
            )

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return best_moves[np.random.randint(len(best_moves))]
