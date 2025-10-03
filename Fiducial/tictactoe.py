import math
import copy

X = "X"
O = "O" 
START_PLAYER = O

# Returns player who has the next turn on a board
def player(board):
    X_count = sum(x.count('X') for x in board)
    O_count = sum(x.count('O') for x in board)

    # If the starting player is X
    if START_PLAYER == X:
        return X if X_count == O_count else O
    else:  # Starting player is O
        return O if X_count == O_count else X


# Returns set of all possible actions (i, j) available on the board
def actions(board):
    actions = set()  # Use a set instead of a list to store unique actions
    
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell != "X" and cell != "O":
                actions.add((i, j))

    return actions

# Returns the board that results from making move (i,j)
def result(board, action):
    # Check if action is valid
    if action not in actions(board):
        raise Exception("Invalid move")
    
    # Make a deep copy of the board, don't want to modify the original
    copy_board = copy.deepcopy(board)
    
    # Place the piece on the board
    copy_board[action[0]][action[1]] = player(board)
    return copy_board

# Returns the winner of the game, if there is one
def winner(board):
                     # Rows
    winning_combos = [[(0, 0), (0, 1), (0, 2)],
                      [(1, 0), (1, 1), (1, 2)],
                      [(2, 0), (2, 1), (2, 2)],
                      # Vertical
                      [(0, 0), (1, 0), (2, 0)],
                      [(0, 1), (1, 1), (2, 1)],
                      [(0, 2), (1, 2), (2, 2)],
                      # Diagonal
                      [(0, 0), (1, 1), (2, 2)],
                      [(0, 2), (1, 1), (2, 0)]]
    
    #New Code for winner function
    for combo in winning_combos:
        if board[combo[0][0]][combo[0][1]] == board[combo[1][0]][combo[1][1]] == board[combo[2][0]][combo[2][1]] != None:
            return board[combo[0][0]][combo[0][1]]
    return None

# Returns True if game is over, False otherwise
def terminal(board):
    # If there is a winner, game is over
    if winner(board) is not None or not actions(board):
        return True
    return False

# Returns 1 if X has won the game, -1 if O has won, 0 otherwise
def utility(board):
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0

#New minimax function
def minimax(board):
    current_player = player(board)
    if current_player == X:
        value, move = maxScore(board, -math.inf, math.inf)
    else:
        value, move = minScore(board, -math.inf, math.inf)
    return move

#New maxScore function
def maxScore(board, alpha, beta):
    if terminal(board):
        return utility(board), None
    bestScore = -math.inf
    optimal_action = None
    for action in actions(board):
        min_result, _ = minScore(result(board, action), alpha, beta)
        if min_result > bestScore:
            bestScore = min_result
            optimal_action = action
        alpha = max(alpha, bestScore)
        if beta <= alpha:
            break
    return bestScore, optimal_action

#New minScore function
def minScore(board, alpha, beta):
    if terminal(board):
        return utility(board), None
    bestScore = math.inf
    optimal_action = None
    for action in actions(board):
        max_result, _ = maxScore(result(board, action), alpha, beta)
        if max_result < bestScore:
            bestScore = max_result
            optimal_action = action
        beta = min(beta, bestScore)
        if beta <= alpha:
            break
    return bestScore, optimal_action

# for debugging
if __name__ == '__main__':
        
    test_board = [
        [None, 'X', 'O'],
        [None, 'O', 'X'],
        [None, None, None]
    ]

    move = minimax(test_board)

    print(move)