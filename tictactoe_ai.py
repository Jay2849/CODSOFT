board = [[' ' for _ in range(3)] for _ in range(3)]

def print_board(current_board):
 
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

def check_draw(board):
   
    for row in board:
        if ' ' in row:
            return False 
    return True 

def minimax(board, depth, is_maximizing):

    if check_winner(board, 'O'):
        return 1
    if check_winner(board, 'X'):
        return -1
    if check_draw(board):
        return 0

    if is_maximizing: 
        best_score = -float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c] == ' ':
                    board[r][c] = 'O'
                    score = minimax(board, depth + 1, False)
                    board[r][c] = ' '
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c] == ' ':
                    board[r][c] = 'X'
                    score = minimax(board, depth + 1, True)
                    board[r][c] = ' '
                    best_score = min(score, best_score)
        return best_score

def ai_move(board):
  
    best_score = -float('inf')
    move = None
    for r in range(3):
        for c in range(3):
            if board[r][c] == ' ':
                board[r][c] = 'O'
                score = minimax(board, 0, False)
                board[r][c] = ' '
                if score > best_score:
                    best_score = score
                    move = (r, c)
    if move:
        board[move[0]][move[1]] = 'O'


print("Welcome to TIC-TAC-TOE! Aap 'X' hain aur AI 'O' hai.")
print_board(board)

while True:
    player_move(board)
    print("\nAapke move ke baad board:")
    print_board(board)

    if check_winner(board, 'X'):
        print("Congratulations! Aap jeet gaye!")
        break
    elif check_draw(board):
        print("Match draw ho gaya!")
        break

    print("\nAI ki baari...")
    ai_move(board)
    print("AI ke move ke baad board:")
    print_board(board)

    if check_winner(board, 'O'):
        print("Sorry, AI jeet gaya!")
        break
    elif check_draw(board):
        print("Match draw ho gaya!")
        break