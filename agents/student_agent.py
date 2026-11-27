"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, LINES, state_key)
from constants import *
import numpy as np
import time

WIN_SCORE = 1000000
MAX_DEPTH = 6
TIME_MARGIN = 0.25


class SearchTimeout(Exception):
    """Stop the unfinished iteration and keep the last complete result."""

@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
        self.turns = 0
        self.completed_depth = 0
        self.nodes = 0
        self.cutoffs = 0
        self.timed_out = False
        self.last_score = None
        self.last_step_seconds = 0.0
        self._deadline = float("inf")
        self._preferred = {}

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
        """Lines where blue has 3 and can complete the 4th next turn."""
        n = 0 # number of threats
        biggest = self.largest_in_reserve(reserves, who)
        for line in LINES[WIN_LENGTH]: # List of every possible winning line on the board
            cells = [owners[r,c] for r,c in line] # owners = who owns top piece on each square
            if OBSTACLE in cells or cells.count(who) != 3: # OBSTACLE = blocked square
                continue
            r,c = next((r,c) for r,c in line if owners[r,c] != who)
            if owners[r,c] == EMPTY:
                n += 1
            elif owners[r,c] == other and sizes[r,c] < biggest: # opponents piece is on top of 4th square and their piece is smaller than the largest pieces we have in reserve 
                n += 1
        return n
    
    def threat_score(self, owners, sizes, reserves, player, opponent):
        mine = self.count_threats(owners, sizes, reserves, player, opponent)
        theirs = self.count_threats(owners, sizes, reserves, opponent, player)
        score = 50 * mine - 60 * theirs # Opponent threats mean more because their threat means we might lose, so blocking is more important
        if mine >= 2: # If 2 or more threats at once, opponent can only block one per turn, so will probably win
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
            below = [board[r,c,s] for s in range(sizes[r,c]) if board[r,c,s] != EMPTY] # List the pieces under your top piece - if large covers opponent's small, below = [2]
            if below and below[-1] == opponent: # Checks if something underneath at all, below[-1] is the biggest piece under yours, if opponents lifting would complete their line
                penalty -= 30 
        return penalty

    def evaluate(self, board, reserves, player, opponent):
        """Overall evaluation: line potential + threats + uncover risk."""
        owners, sizes = top_view(board)
        return (self.line_potential(board, player, opponent)
                + self.threat_score(owners, sizes, reserves, player, opponent)
                + self.uncover_risk(board, owners, sizes, player, opponent))

    def _check_time(self):
        if time.perf_counter() >= self._deadline:
            raise SearchTimeout

    def _ordered_moves(self, board, moves, player, preferred=None):
        """Try the previous best move, covers, then placements first.
        This changes only the visit order. Every move remains eligible.
        """
        owners, _ = top_view(board)
        return sorted(moves, key=lambda move: (
            move == preferred,
            owners[move.dest] == 3 - player,
            move.src is None,
            move.size,
        ), reverse=True)

    def minimax(self, board, reserves, depth, alpha, beta,
                current_player, root_player, ply=0):
        """Alpha-beta search with deadline and official endgame precedence."""
        self._check_time()
        self.nodes += 1
        winner = check_winner(board, 3 - current_player, WIN_LENGTH)
        if winner:
            return WIN_SCORE + depth if winner == root_player else -WIN_SCORE - depth
        # A line on the last allowed move still wins before the draw check.
        if ply >= MAX_PLIES:
            return 0

        moves = get_valid_moves(board, reserves, current_player, ALLOW_MOVES)
        if not moves:
            if ALLOW_MOVES:
                return (-WIN_SCORE - depth if current_player == root_player
                        else WIN_SCORE + depth)
            if not get_valid_moves(board, reserves, 3 - current_player, ALLOW_MOVES):
                return 0
            return self.minimax(board, reserves, max(0, depth - 1), alpha, beta,
                                3 - current_player, root_player, ply + 1)
        if depth == 0:
            return self.evaluate(board, reserves, root_player, 3 - root_player)

        key = state_key(board, reserves, current_player)
        moves = self._ordered_moves(board, moves, current_player,
                                    self._preferred.get(key))
        maximizing = current_player == root_player
        value = float("-inf") if maximizing else float("inf")
        best_move = moves[0]
        for move in moves:
            self._check_time()
            child_board, child_reserves = copy_state(board, reserves)
            execute_move(child_board, child_reserves, current_player, move, ALLOW_MOVES)
            score = self.minimax(child_board, child_reserves, depth - 1,
                                 alpha, beta, 3 - current_player, root_player, ply + 1)
            if (maximizing and score > value) or (not maximizing and score < value):
                value, best_move = score, move
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                self.cutoffs += 1
                break
        # These are move-order hints only, never cached score bounds.
        if len(self._preferred) < 10000:
            self._preferred[key] = best_move
        return value

    def _search_root(self, board, reserves, player, moves, depth, ply):
        alpha = float("-inf")
        best_move = moves[0]
        scored_moves = []
        for move in moves:
            self._check_time()
            child_board, child_reserves = copy_state(board, reserves)
            execute_move(child_board, child_reserves, player, move, ALLOW_MOVES)
            score = self.minimax(child_board, child_reserves, depth - 1,
                                 alpha, float("inf"), 3 - player, player, ply + 1)
            scored_moves.append((score, move))
            # Equality can be a fail-low bound, not an exact tie. Keep the
            # incumbent unless a move strictly improves its established value.
            if score > alpha:
                alpha, best_move = score, move
            if alpha >= WIN_SCORE + depth - 1:
                break  # immediate win: no alternative can score higher
        scored_moves.sort(key=lambda item: item[0], reverse=True)
        ordered = [move for _, move in scored_moves]
        ordered.extend(move for move in moves if move not in ordered)
        return best_move, alpha, ordered

    def step(self, board, reserves, player, opponent):
        """Search progressively deeper and return the last finished result."""
        start = time.perf_counter()
        self._deadline = start + max(0.0, STEP_TIME_LIMIT - TIME_MARGIN)
        self.nodes = 0
        self.cutoffs = 0
        self.completed_depth = 0
        self.timed_out = False
        self.last_score = None
        self._preferred = {}
        # With board moves enabled, the framework calls us on every turn.
        # The placement-only variant ends before the 100-ply cap matters.
        ply = 2 * self.turns + player - 1 if ALLOW_MOVES else 0
        self.turns += 1
        moves = get_valid_moves(board, reserves, player, ALLOW_MOVES)
        best_move = moves[0]  # the framework calls step only with legal moves
        try:
            for depth in range(1, MAX_DEPTH + 1):
                move, score, ordered = self._search_root(
                    board, reserves, player, moves, depth, ply)
                # Commit only a complete iteration. A timeout cannot replace it
                # with an unfair comparison of partially explored candidates.
                best_move, moves = move, ordered
                self.completed_depth = depth
                self.last_score = score
                if abs(score) >= WIN_SCORE:
                    break
        except SearchTimeout:
            self.timed_out = True
        self.last_step_seconds = time.perf_counter() - start
        return best_move