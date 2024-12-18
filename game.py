import numpy as np
import itertools
from typing import List, Tuple, Optional


class CubicGame:
    def __init__(self):
        # 4x4x4 grid representing the game board
        self.board = np.zeros((4, 4, 4), dtype=int)
        self.current_player = 1  # 1 for player, -1 for AI
        self.game_over = False
        self.winner = None
        

    def make_move(self, x: int, y: int, z: int) -> bool:
        """
        Place a marker on the board for the current player.
        """
        if self.board[x, y, z] == 0 and not self.game_over:
            self.board[x, y, z] = self.current_player
            self.check_winner()
            self.current_player *= -1  # Switch player
            return True
        return False

    def check_winner(self) -> Optional[int]:
        """
        Check for a winner in all possible directions.
        """
        directions = self._generate_check_directions()

        for direction in directions:
            for start in itertools.product(range(4), repeat=3):
                line = []
                for j in range(4):
                    pos = tuple(start[i] + direction[i] * j for i in range(3))
                    if not all(0 <= pos[i] < 4 for i in range(3)):  # Ensure position is valid
                        break
                    line.append(self.board[pos])

                if len(line) == 4 and len(set(line)) == 1 and line[0] != 0:
                    self.game_over = True
                    self.winner = line[0]
                    return self.winner

        # Check for draw
        if np.all(self.board != 0):
            self.game_over = True
            self.winner = 0  # Indicate a draw
        return None


    def _generate_check_directions(self) -> List[Tuple[int, int, int]]:
        """
        Generate all possible checking directions in 3D space.
        """
        directions = []

        # Horizontal, vertical, and depth directions
        axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        directions.extend(axes)

        # Diagonal lines across each plane
        diagonals = [
            (1, 1, 0), (1, -1, 0),  # XY-plane
            (1, 0, 1), (1, 0, -1),  # XZ-plane
            (0, 1, 1), (0, 1, -1)   # YZ-plane
        ]
        directions.extend(diagonals)

        # Cube diagonals
        cube_diagonals = [
            (1, 1, 1), (1, 1, -1),
            (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1),
            (-1, -1, 1), (-1, -1, -1)
        ]
        directions.extend(cube_diagonals)

        return directions


class CubicAI:
    def __init__(self, player: int):
        self.player = player
        self.depth = 2  # Reduced depth for faster moves


    def heuristic1(self, game: CubicGame) -> float:
        """
        Heuristic 1: Line potential evaluation.
        """
        score = 0
        directions = game._generate_check_directions()

        for direction in directions:
            for start in itertools.product(range(4), repeat=3):
                line = []
                for j in range(4):
                    pos = tuple(start[i] + direction[i] * j for i in range(3))
                    if not all(0 <= pos[i] < 4 for i in range(3)):
                        break
                    line.append(game.board[pos])

                if len(line) == 4:
                    ai_count = line.count(self.player)
                    player_count = line.count(-self.player)

                    if ai_count > 0 and player_count == 0:
                        score += 10 ** ai_count  # Favor lines with more AI pieces
                    elif player_count > 0 and ai_count == 0:
                        score -= 10 ** player_count  # Penalize lines with opponent pieces
        return score


    def heuristic2(self, game: CubicGame) -> float:
        """
        Heuristic 2: Threat-based evaluation.
        """
        score = 0
        center = np.array([1.5, 1.5, 1.5])  # Approximate center of the board

        # Check for immediate winning or losing conditions first
        if game.winner == self.player:
            return float('inf')
        elif game.winner == -self.player:
            return float('-inf')

        # Comprehensive threat detection
        def analyze_lines(board, player):
            threat_score = 0
            directions = game._generate_check_directions()

            for direction in directions:
                for start in itertools.product(range(4), repeat=3):
                    line = []
                    valid_line = True
                    for j in range(4):
                        pos = tuple(start[i] + direction[i] * j for i in range(3))
                        if not all(0 <= pos[i] < 4 for i in range(3)):
                            valid_line = False
                            break
                        line.append(board[pos])

                    if not valid_line:
                        continue

                    # Extremely high priority for blocking/creating 3-in-a-row
                    player_count = line.count(player)
                    opponent_count = line.count(-player)
                    empty_count = line.count(0)

                    # Highest priority: Prevent immediate loss or secure immediate win
                    if player_count == 3 and empty_count == 1:
                        threat_score += 1000 if player == self.player else -1000
                    
                    # High priority: Block 2-in-a-row setups or create own 2-in-a-row
                    elif player_count == 2 and empty_count == 2:
                        threat_score += 100 if player == self.player else -100
                    
                    # Medium priority: Partial line control
                    elif player_count == 2 and opponent_count == 1 and empty_count == 1:
                        threat_score += 50 if player == self.player else -50

            return threat_score

        # Evaluate board state
        threat_evaluation = analyze_lines(game.board, self.player)
        score += threat_evaluation

        # Positional scoring
        for x, y, z in itertools.product(range(4), repeat=3):
            if game.board[x, y, z] == self.player:
                # Reward proximity to center and control of strategic positions
                distance_to_center = np.linalg.norm(np.array([x, y, z]) - center)
                score += max(1, 4 - distance_to_center) * 5
            
            elif game.board[x, y, z] == -self.player:
                # Penalize opponent's proximity and control
                distance_to_center = np.linalg.norm(np.array([x, y, z]) - center)
                score -= max(1, 4 - distance_to_center) * 5

        return score


    def minimax(self, game: CubicGame, depth: int, alpha: float, beta: float, maximizing_player: bool, heuristic, use_alpha_beta: bool) -> Tuple[float, Optional[Tuple[int, int, int]]]:
        """
        Minimax algorithm with optional Alpha-Beta pruning and heuristic function.
        """
        if depth == 0 or game.game_over:
            if heuristic:
                return heuristic(game), None  # Use the provided heuristic function
            else:
                return self._evaluate_board(game), None  # Use default evaluation if no heuristic is provided

        empty_positions = [(x, y, z) for x, y, z in itertools.product(range(4), repeat=3)
                        if game.board[x, y, z] == 0]

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in empty_positions:
                game_copy = self._copy_game_state(game)
                game_copy.make_move(*move)
                eval, _ = self.minimax(game_copy, depth - 1, alpha, beta, False, heuristic, use_alpha_beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                if use_alpha_beta:
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval, best_move

        else:
            min_eval = float('inf')
            best_move = None
            for move in empty_positions:
                game_copy = self._copy_game_state(game)
                game_copy.make_move(*move)
                eval, _ = self.minimax(game_copy, depth - 1, alpha, beta, True, heuristic, use_alpha_beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                if use_alpha_beta:
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval, best_move


    def _evaluate_board(self, game: CubicGame) -> float:
        """
        Heuristic function to evaluate the board state.
        """
        score = 0
        directions = game._generate_check_directions()

        for direction in directions:
            for start in itertools.product(range(4), repeat=3):
                line = []
                for j in range(4):
                    pos = tuple(start[i] + direction[i] * j for i in range(3))
                    if not all(0 <= pos[i] < 4 for i in range(3)):
                        break
                    line.append(game.board[pos])

                if len(line) == 4:
                    ai_count = line.count(self.player)
                    player_count = line.count(-self.player)

                    if ai_count == 4:
                        return float('inf')  # Immediate win
                    if player_count == 4:
                        return float('-inf')  # Immediate loss

                    # Adjust score based on potential
                    if ai_count > 0 and player_count == 0:
                        score += 10 ** ai_count
                    if player_count > 0 and ai_count == 0:
                        score -= 10 ** player_count
        return score


    def _copy_game_state(self, game: CubicGame) -> CubicGame:
        """
        Create a deep copy of the game state.
        """
        new_game = CubicGame()
        new_game.board = game.board.copy()
        new_game.current_player = game.current_player
        new_game.game_over = game.game_over
        new_game.winner = game.winner
        return new_game


    def get_best_move(self, game: CubicGame) -> Optional[Tuple[int, int, int]]:
        """
        Get the best move for the AI player.
        """
        _, best_move = self.minimax(
            game,
            self.depth,
            float('-inf'),
            float('inf'),
            True,
            self.selected_heuristic,
            use_alpha_beta=self.use_alpha_beta
        )
        return best_move


    def test_heuristic(self, game: CubicGame, heuristic) -> Tuple[Optional[Tuple[int, int, int]], float]:
        """
        Test a heuristic with Minimax and Alpha-Beta pruning.
        """
        import time
        start_time = time.time()
        
        # Pass use_alpha_beta when calling minimax
        _, best_move = self.minimax(game, self.depth, float('-inf'), float('inf'), True, heuristic, self.use_alpha_beta)
        
        end_time = time.time()

        # Handle the case where heuristic is None
        heuristic_name = heuristic.__name__ if heuristic else "None"
        print(f"Heuristic: {heuristic_name}, Best Move: {best_move}, Time Taken: {end_time - start_time:.4f}s")
        return best_move, end_time - start_time


