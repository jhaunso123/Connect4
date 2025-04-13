import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

import pygame
import cv2
import mediapipe as mp
import math
import threading


"""## Integrate the neural network"""

class ConnectFourNN:
    def __init__(self):
        self.grid = np.zeros((6, 7), dtype=int)
        self.current_player = 1

    def make_move(self, col):
        if col < 0 or col >= self.grid.shape[1]:
            print("Invalid column. Please choose a column between 0 and 6.")
            return False

        for i in range(5, -1, -1):
            if self.grid[i][col] == 0:
                self.grid[i][col] = self.current_player
                return True

        print(f"Column {col} is full. Choose another column.")
        return False

    def show_board(self):
        print("\n 0  1  2  3  4  5  6")
        print("+--+--+--+--+--+--+--+")
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
        rows, cols = self.grid.shape
        for i in range(rows):
            for j in range(cols - 3):
                if all(self.grid[i][j + k] == self.current_player for k in range(4)):
                    return True

        for j in range(cols):
            for i in range(rows - 3):
                if all(self.grid[i + k][j] == self.current_player for k in range(4)):
                    return True

        for i in range(rows - 3):
            for j in range(cols - 3):
                if all(self.grid[i + k][j + k] == self.current_player for k in range(4)):
                    return True

        for i in range(rows - 3):
            for j in range(3, cols):
                if all(self.grid[i + k][j - k] == self.current_player for k in range(4)):
                    return True

        return False

    def is_draw(self):
        return all(self.grid[0][i] != 0 for i in range(7))

    def get_possible_moves(self):
        return [i for i in range(7) if self.grid[0][i] == 0]

    def clone(self):
        clone_game = ConnectFour()
        clone_game.grid = np.copy(self.grid)
        clone_game.current_player = self.current_player
        return clone_game

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def get_nn_move(self, model):
        board_state = self.grid.reshape((1, 6, 7, 1))  # Reshape for the neural network input
        probabilities = model.predict(board_state)[0]
        valid_moves = self.get_possible_moves()
        best_move = max(valid_moves, key=lambda move: probabilities[move])
        return best_move
    
def draw_board(game, selected_col, screen, SQUARESIZE, RADIUS, height, myfont):
    board = game.grid
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    ROW_COUNT, COLUMN_COUNT = 6, 7
    width = COLUMN_COUNT * SQUARESIZE

    # Draw the board background
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            # Flip the row to draw bottom-up
            draw_r = r
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, draw_r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
                int(c * SQUARESIZE + SQUARESIZE / 2),
                int(draw_r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)
            ), RADIUS)

    # Draw the pieces
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            cell = board[r][c]
            if cell != 0:
                # Flip the row index to draw from the bottom
                draw_r = ROW_COUNT - 1 - r
                color = RED if cell == 1 else YELLOW
                pygame.draw.circle(screen, color, (
                    int(c * SQUARESIZE + SQUARESIZE / 2),
                    int(draw_r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)
                ), RADIUS)

    # Draw the selector
    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
    color = RED if game.current_player == 1 else YELLOW
    pygame.draw.circle(screen, color, (
        int(selected_col * SQUARESIZE + SQUARESIZE / 2),
        int(SQUARESIZE / 2)
    ), RADIUS)

    pygame.display.update()





def head_control():
    global selected_col, move_confirmed, running

    COLUMN_COUNT = 7
    TILT_THRESHOLD = 10
    prev_direction = "Centered"

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        h, w, _ = frame.shape

        direction = "Centered"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
                right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))
                dx = right_eye_coords[0] - left_eye_coords[0]
                dy = right_eye_coords[1] - left_eye_coords[1]
                angle = math.degrees(math.atan2(dy, dx))

                if angle > TILT_THRESHOLD:
                    direction = "Right"
                elif angle < -TILT_THRESHOLD:
                    direction = "Left"

        # ✅ Only update if direction changed from previous frame
        if direction != prev_direction:
            if direction == "Right" and selected_col < COLUMN_COUNT - 1:
                selected_col += 1
                print("Moved Right ->", selected_col)
            elif direction == "Left" and selected_col > 0:
                selected_col -= 1
                print("Moved Left ->", selected_col)
            prev_direction = direction

        key = cv2.waitKey(1)
        if key == 27:
            running = False
         

    cap.release()


###################################################################### GLOBAL VARIABLES BELOW ##########################################

selected_col = 3  # Start in the center column
move_confirmed = False
running = True



def play_game():

    global selected_col, move_confirmed, running

    game = ConnectFourNN()
    game.current_player = 1  # Ensure human goes first

    model = tf.keras.models.load_model('neural_network/connect4_model.keras')

    pygame.init()
    SQUARESIZE = 100
    RADIUS = int(SQUARESIZE / 2 - 5)
    width = 7 * SQUARESIZE
    height = (6 + 1) * SQUARESIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Connect 4 - Head Tilt Controlled")
    myfont = pygame.font.SysFont("monospace", 75)

    threading.Thread(target=head_control, daemon=True).start()

    draw_board(game, selected_col, screen, SQUARESIZE, RADIUS, height, myfont)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and game.current_player == 1:
                    print("SPACEBAR pressed in Pygame loop")
                    move_confirmed = True

        #print("Current Player:", game.current_player)
        #print("Selected Col:", selected_col)
        #print("Move Confirmed:", move_confirmed)

        if move_confirmed and game.current_player == 1:
            print("Trying to place piece for Player 1...")
            move_confirmed = False

            if selected_col in game.get_possible_moves():
                game.make_move(selected_col)
                print("GRID AFTER MOVE:")
                print(game.grid)
                print("Piece placed!")
                draw_board(game, selected_col, screen, SQUARESIZE, RADIUS, height, myfont)

                if game.check_win():
                    label = myfont.render("You Win!", 1, (255, 0, 0))
                    screen.blit(label, (40, 10))
                    pygame.display.update()
                    pygame.time.wait(3000)
                    break
                elif game.is_draw():
                    label = myfont.render("Draw!", 1, (255, 255, 0))
                    screen.blit(label, (40, 10))
                    pygame.display.update()
                    pygame.time.wait(3000)
                    break

                game.switch_player()
            else:
                print("Invalid move — column is full.")

        elif game.current_player == 2:
            print("AI turn...")
            pygame.time.wait(500)
            ai_col = game.get_nn_move(model)
            if ai_col in game.get_possible_moves():
                game.make_move(ai_col)
                draw_board(game, selected_col, screen, SQUARESIZE, RADIUS, height, myfont)

                if game.check_win():
                    label = myfont.render("AI Wins!", 1, (255, 255, 0))
                    screen.blit(label, (40, 10))
                    pygame.display.update()
                    pygame.time.wait(3000)
                    break
                elif game.is_draw():
                    label = myfont.render("Draw!", 1, (255, 255, 0))
                    screen.blit(label, (40, 10))
                    pygame.display.update()
                    pygame.time.wait(3000)
                    break

                game.switch_player()

        draw_board(game, selected_col, screen, SQUARESIZE, RADIUS, height, myfont)

    pygame.quit()


if __name__ == "__main__":
    play_game()