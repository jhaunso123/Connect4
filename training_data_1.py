# Import necessary libraries
import numpy as np  # Import NumPy library for numerical operations
import random  # Import random library for generating random numbers
import pickle  # Import pickle library for serializing and deserializing objects
import argparse

iteration_start = 0

# Define ConnectFour class
class ConnectFour:
    def __init__(self):
        # Initialize the game board with zeros (6 rows and 7 columns)
        self.grid = np.zeros((6, 7), dtype=int)
        # Set the current player to player 1
        self.current_player = 1

    def make_move(self, col):
        # Check if the chosen column is valid
        if col < 0 or col >= self.grid.shape[1]:
            print("Invalid column. Please choose a column between 0 and 6.")
            return False

        # Place the player's piece in the chosen column
        for i in range(5, -1, -1):
            if self.grid[i][col] == 0:
                self.grid[i][col] = self.current_player
                return True

        # If the column is full, print a message and return False
        print(f"Column {col} is full. Choose another column.")
        return False

    def show_board(self):
        # Print the column numbers
        print("\n 0  1  2  3  4  5  6")
        print("+--+--+--+--+--+--+--+")
        # Print the current state of the game board
        for row in self.grid:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print(" . ", end="")
                elif cell == 1:
                    print(" X ", end="")
                else:
                    print(" O ", end="")
            print("|")
        print("+--+--+--+--+--+--+--+")

    def check_win(self):
        # Check for a winning sequence of four in a row
        rows, cols = self.grid.shape
        for i in range(rows):
            for j in range(cols - 3):
                if all(self.grid[i][j + k] == self.current_player for k in range(4)):
                    return True

        # Check for a winning sequence of four in a column
        for j in range(cols):
            for i in range(rows - 3):
                if all(self.grid[i + k][j] == self.current_player for k in range(4)):
                    return True

        # Check for a winning sequence of four diagonally (top-left to bottom-right)
        for i in range(rows - 3):
            for j in range(cols - 3):
                if all(self.grid[i + k][j + k] == self.current_player for k in range(4)):
                    return True

        # Check for a winning sequence of four diagonally (top-right to bottom-left)
        for i in range(rows - 3):
            for j in range(3, cols):
                if all(self.grid[i + k][j - k] == self.current_player for k in range(4)):
                    return True

        return False

    def is_draw(self):
        # Check if the game is a draw (no empty spaces in the top row)
        return all(self.grid[0][i] != 0 for i in range(7))

    def get_possible_moves(self):
        # Get a list of columns that are not full
        return [i for i in range(7) if self.grid[0][i] == 0]

    def clone(self):
        # Create a copy of the current game state
        clone_game = ConnectFour()
        clone_game.grid = np.copy(self.grid)
        clone_game.current_player = self.current_player
        return clone_game

    def switch_player(self):
        # Switch the current player
        self.current_player = 3 - self.current_player

# Define MCTSNode class for Monte Carlo Tree Search
class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        # Clone the current game state
        self.game = game.clone()
        self.parent = parent  # Set the parent node
        self.move = move  # Set the move that led to this node
        self.wins = 0  # Initialize wins count
        self.winsScore = 0  # Initialize wins count
        self.losses = 0  # Initialize losses count
        self.lossesScore = 0  # Initialize losses count
        self.draws = 0  # Initialize draws count
        self.visits = 0  # Initialize visits count
        self.children = []  # Initialize list of child nodes
        self.untried_moves = game.get_possible_moves()  # Get possible moves from the current game state

    def select_child(self):
        # Select the child node with the highest UCT (Upper Confidence Bound for Trees) value.
        # UCT value is calculated using the formula:
        # UCT = (w_i / n_i) + sqrt(2 * log(N) / n_i)
        # where:
        # - w_i is the number of wins for the child node i
        # - n_i is the number of visits for the child node i
        # - N is the total number of visits for the current node (parent node)
        #
        # This formula balances exploration and exploitation by considering both the success rate
        # of the child node (w_i / n_i) and the potential for unexplored nodes (sqrt(2 * log(N) / n_i)).
        #
        # `sorted(self.children, key=lambda c: ...)` sorts the list of child nodes based on their UCT values.
        # `lambda c: c.wins / c.visits + np.sqrt(2 * np.log(self.visits) / c.visits)` is a lambda function that computes
        # the UCT value for each child node.
        #
        # The `sorted` function returns a list of child nodes sorted in ascending order based on their UCT values.
        # The `[-1]` index selects the last element from the sorted list, which corresponds to the child node
        # with the highest UCT value.
        
        # return sorted(self.children, key=lambda c: c.wins / c.visits + np.sqrt(2 * np.log(self.visits) / c.visits))[-1]

        return sorted(self.children, key=lambda c: random.random())[-1]


    def expand(self):
        # Expand the node by creating a new child node
        move = self.untried_moves.pop()
        next_game = self.game.clone()
        next_game.make_move(move)
        next_game.switch_player()
        child_node = MCTSNode(next_game, self, move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        # Update the node with the result of a simulation
        self.visits += 1
        if result > 0:
            self.wins += 1
            self.winsScore += result
        if result < 0:
            self.losses += 1
            self.lossesScore += (0 - result)
        if result == 0:
            self.draws += 1

# Define MCTS function for Monte Carlo Tree Search
def MCTS(root, itermax):
    # Get the player using MCTS
    mcts_player = root.game.current_player + 0

    for _ in range(itermax):
        node = root
        game = root.game.clone()

        #print(f"ITERATION {_}")

        # Select phase
        while node.untried_moves == [] and node.children != []:
            level_simulation = 1
            #print("SELECT")
            node = node.select_child()
            game.make_move(node.move)
            game.switch_player()

        # Expand phase
        if node.untried_moves != []:
            level_simulation = 1
            #print("EXPAND")
            node = node.expand()
            #print(f"NODE MOVE: {node.move}")
            game.make_move(node.move)
            game.switch_player()

        # Simulate phase
        while game.get_possible_moves() != [] and not game.check_win():
            level_simulation += 1
            #print(f"SIMULATE - LEVEL {level_simulation}")
            game.make_move(random.choice(game.get_possible_moves()))
            game.switch_player()

        # Backpropagate phase
        while node != None:
            if game.check_win():
                if game.current_player != mcts_player:
                    result = -1  # The opponent won
                else:
                    result = 1  # The AI won
            else:
                result = 0  # Draw

            # if level_simulation == 1 and result == 1:
            #     node.update(result * 1000)
            # elif level_simulation == 2 and result == -1:
            #     node.update(result * 3000)
            # elif level_simulation >= 1:
            #     node.update(result/(level_simulation+1))

            if level_simulation >= 1:
                node.update(result/(level_simulation+1))

            node = node.parent

# Define function to generate training data
def generate_training_data(num_games, itermax):
    training_data = []  # Initialize list to store training data
    save_interval = 2  # Set the interval to save the training data

    for i in range(num_games):
        #random.seed(i+iteration_start)  # Set the random seed for reproducibility
        game = ConnectFour()  # Create a new ConnectFour game

        print("============ INITIAL STATE ==============")
        game.show_board()
        print("=========================================")        

        while True:
            print(f"Training data: {len(training_data)} states - {i+1} out of {num_games} completed")
            print(f"Current player: {game.current_player}")
            board_state = game.grid.copy()  # Copy the current game board
            root = MCTSNode(game)  # Create a new MCTSNode
            MCTS(root, itermax + (i * 100))  # Run MCTS with increasing iterations
            sorted_list = sorted(root.children, key=lambda c: (c.winsScore-c.lossesScore)) # Sorted list

            # for _ in sorted_list:
            #     print(f"Move: {_.move}")
            #     print(f"Visits: {_.visits}")
            #     print(f"Wins: {_.wins}")
            #     print(f"Wins score: {_.winsScore}")
            #     print(f"Losses: {_.losses}")
            #     print(f"Losses score: {_.lossesScore}\n")


            col = sorted_list[-1].move  # Get the move with the most visits
            training_data.append((board_state, col))  # Append the board state and move to training data
            game.make_move(col)  # Make the move on the game board
            game.show_board()  # Show the current game board
            if game.check_win() or game.is_draw():
                break  # End the game if there is a win or a draw
            game.switch_player()  # Switch the current player

        # Save the training data every 100 games
        if (i + 1) % save_interval == 0:
            with open(f'training_data/training_data_{itermax}_{i+1}.pkl', 'wb') as f:
                pickle.dump(training_data, f)  # Save the training data to a file
            training_data = []  # Reset training data for the next interval

    # Save any remaining training data
    if training_data:
        with open(f'training_data/training_data_{itermax}_{num_games}.pkl', 'wb') as f:
            pickle.dump(training_data, f)  # Save the remaining training data to a file

def main():
    parser = argparse.ArgumentParser(description="My script")
    parser.add_argument("integer_arg", type=int, help="Iterations")
    # args = parser.parse_args()

    # iteration_start = args.integer_arg

    iteration_start = 6000

    print(f"Iteration start: {iteration_start}")

    generate_training_data(10, iteration_start)



if __name__ == "__main__":
    main()