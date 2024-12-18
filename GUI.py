import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from game import CubicGame, CubicAI


class CubicGameGUI:
    def __init__(self, master, ai):
        self.master = master
        master.title("3D Cubic Game")

        # Game initialization
        self.game = CubicGame()
        self.ai = ai  # AI plays as -1

        # Create GUI elements
        self._create_board_view()
        self._create_status_label()

    def _create_board_view(self):
        """
        Create a 4x4x4 grid representation of the game board in a 4x1 column, with reduced size.
        """
        self.buttons = {}
        self.board_frame = tk.Frame(self.master)
        self.board_frame.pack(pady=5)  # Reduce outer padding

        # Create a 4x1 column of layers
        for layer_z in range(4):
            layer_frame = tk.Frame(self.board_frame, relief=tk.RAISED, borderwidth=2)
            layer_frame.pack(side=tk.TOP, pady=3)  # Reduced vertical padding between layers

            # Layer label
            tk.Label(layer_frame, text=f'Layer {layer_z}', font=('Arial', 8)).pack(pady=1)  # Smaller font and padding

            # Create 4x4 grid for this layer
            for y in range(4):
                row_frame = tk.Frame(layer_frame)
                row_frame.pack(side=tk.TOP)
                for x in range(4):
                    btn = tk.Button(row_frame, text='', width=2, height=1,  # Smaller button size
                                    command=lambda row=x, col=y, layer=layer_z:
                                    self._on_button_click(row, col, layer))
                    btn.pack(side=tk.LEFT, padx=1, pady=1)  # Reduced inner padding
                    self.buttons[(x, y, layer_z)] = btn


    def _create_status_label(self):
        """
        Create status label for game information.
        """
        self.status_label = tk.Label(self.master, text="Player's Turn", font=('Arial', 12))
        self.status_label.pack(pady=10)

    def _update_board_view(self):
        """
        Update button states for all layers.
        """
        for (x, y, z), btn in self.buttons.items():
            board_value = self.game.board[x, y, z]
            if board_value == 1:
                btn.config(text='X', state=tk.DISABLED, bg='lightblue')
            elif board_value == -1:
                btn.config(text='O', state=tk.DISABLED, bg='lightcoral')
            else:
                btn.config(text='', state=tk.NORMAL, bg='SystemButtonFace')

    def _on_button_click(self, x, y, z):
        """
        Handle player's move.
        """
        if self.game.current_player != 1:
            return

        if self.game.make_move(x, y, z):
            self._update_board_view()

            if self.game.game_over:
                self._handle_game_end()
                return

            self._ai_move()

    def _ai_move(self):
        """
        Execute AI's move.
        """
        if self.game.current_player != -1:
            return

        # Call test_heuristic to get the best move and time taken
        best_move, time_taken = self.ai.test_heuristic(self.game, self.ai.selected_heuristic)

        if best_move:
            x, y, z = best_move
            self.game.make_move(x, y, z)
            self._update_board_view()

            if self.game.game_over:
                self._handle_game_end()

        # Optionally, update status with heuristic time info
        self.status_label.config(text=f"AI Move Taken! Time: {time_taken:.4f}s")

    def _handle_game_end(self):
        """
        Handle game end scenarios.
        """
        if self.game.winner == 1:
            messagebox.showinfo("Game Over", "Player Wins!")
        elif self.game.winner == -1:
            messagebox.showinfo("Game Over", "AI Wins!")
        else:
            messagebox.showinfo("Game Over", "It's a Draw!")

        self.game = CubicGame()
        self._update_board_view()


def main():
    root = tk.Tk()
    root.title("Cubic Game Settings")

    ai = CubicAI(player=-1)


    def start_game():
        # Get user selections
        algorithm = algorithm_var.get()
        heuristic = heuristic_var.get()

        # Map heuristic choice to actual function
        if heuristic == "Heuristic 1":
            selected_heuristic = ai.heuristic1
        elif heuristic == "Heuristic 2":
            selected_heuristic = ai.heuristic2
        elif heuristic == "None":
            selected_heuristic = None  # No heuristic, AI will only use the algorithm
        else:
            messagebox.showerror("Error", "Please select a valid heuristic!")
            return

        # Configure AI
        ai.use_alpha_beta = (algorithm == "Alpha-Beta Pruning")
        ai.selected_heuristic = selected_heuristic

        # Start the game
        game_window = tk.Toplevel(root)
        CubicGameGUI(game_window, ai)


    # Algorithm selection
    tk.Label(root, text="Choose Algorithm:", font=("Arial", 12)).pack(pady=5)
    algorithm_var = tk.StringVar(value="Minimax")
    algorithm_menu = ttk.Combobox(root, textvariable=algorithm_var, state="readonly")
    algorithm_menu['values'] = ["Minimax", "Alpha-Beta Pruning"]
    algorithm_menu.pack(pady=5)

    # Heuristic selection
    tk.Label(root, text="Choose Heuristic Function:", font=("Arial", 12)).pack(pady=5)
    heuristic_var = tk.StringVar(value="None")  # Default to "None"
    heuristic_menu = ttk.Combobox(root, textvariable=heuristic_var, state="readonly")
    heuristic_menu['values'] = ["None", "Heuristic 1", "Heuristic 2"]  # Add "None" as an option
    heuristic_menu.pack(pady=5)

    # Start button
    tk.Button(root, text="Start Game", font=("Arial", 12), command=start_game).pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    main()