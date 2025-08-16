board = [[' ' for _ in range(3)] for _ in range(3)]

def print_board(current_board):
    """
    Yeh function game board ko terminal mein print karta hai.
    """
    print("-------------")
    for row in current_board:
        print(f"| {row[0]} | {row[1]} | {row[2]} |")
        print("-------------")


print("Welcome to TIC-TAC-TOE!")
print_board(board)

def player_move(board):

    while True:
        try:
            
            row = int(input("Apna 'X' lagane ke liye row chune (0, 1, or 2): "))
            col = int(input("Apna 'X' lagane ke liye column chune (0, 1, or 2): "))

            if 0 <= row <= 2 and 0 <= col <= 2:
                if board[row][col] == ' ':
                    board[row][col] = 'X'
                    break
                else:
                    print("Yeh spot pehle se bhara hua hai. Koi aur spot chune.")
            else:
                print("Galat input! Sirf 0, 1, ya 2 chune.")
        except ValueError:
            print("Yeh ek number nahi hai! Sirf 0, 1, ya 2 dalein.")

def check_winner(board, player):

    win_conditions = [
        # Rows
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        # Columns
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        # Diagonals
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]]
    ]

    for condition in win_conditions:
        if all(cell == player for cell in condition):
            return True
    return False

print("Welcome to TIC-TAC-TOE!")

current_player = 'X'

for _ in range(5):
    print_board(board)
  
    player_move(board)
    
    if check_winner(board, current_player):
        print("\nCongratulations! Player 'X' jeet gaya!")
        print_board(board) 
        break