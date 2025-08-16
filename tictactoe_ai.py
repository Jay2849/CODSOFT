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