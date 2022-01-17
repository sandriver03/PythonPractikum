"""
python script for the Rock-Paper-Scissor game
"""

import random


# allowed choices
allowed_vals = ['Rock', 'Paper', 'Scissor']

# to run the Rock-Paper-Scissor game (player vs. computer):
# we want to ask if the user want to play multiple game
# if we know how to play one game, then play multiple game is not a problem
# while (want to play):
    # play one game

# to play one game
# get player choice

# generate computer choice
# compare player choice and computer choice


def get_player_choice(allowed_vals):
    """

    :param allowed_vals:
    :return:
    """



# player choice
def get_player_choice(allowed_vals):
    """
    ask the user to type in a choice from the Python interpreter, and return the value typed
    :inputs:
        allowed_vals: list of strings, values allowed to be typed in
    :return:
        string, the choice of the player
    """
    # input allows user to type in any thing they want, how can we limit the input?
    choice = ''
    # we should tell the user which values are allowed
    # ToDO
    while choice not in allowed_vals:
        choice = input('Enter your choice: ')
    return choice


# computer choice
def generate_computer_choice(allowed_vals):
    """
    randomly pick one value from the allowed_vals list
    :param allowed_vals: list
    :return: single element from the allowed_vals list
    """
    return random.choice(allowed_vals)


# compare choices
def compare_choices(player_choice, computer_choice):
    """
    compare the two choices and return the result (draw, win, lose)
    :param player_choice: string
    :param computer_choice: string
    :return: string, in ('d', 'w', 'l')
    """
    # ToDO
    # choices have to be in ('Rock', 'Paper', 'Scissor')
    # we can use a lot of if ... elif to do the comparison
    # or, we can set a value to each choice
    choice_values = {'Rock': 0, 'Paper': 1, 'Scissor': 2}
    # in this case, take the difference between the values of the two choice, and the difference is periodic
    diff_val = choice_values[player_choice] - choice_values[computer_choice]
    # how to map to draw, win, loss from the diff_val?
    # 0 is draw
    if diff_val == 0:
        return 'd'
    # -1 and 2 is loss, or in general diff_val % 3 == 2 is loss
    elif diff_val % 3 == 2:
        return 'l'
    # 1 and -2 is win, or diff_val % 3 == 1 is win
    elif diff_val % 3 == 1:
        return 'w'


def play_one_game(allowed_vals):
    player_choice = get_player_choice(allowed_vals)
    computer_choice = generate_computer_choice(allowed_vals)
    return compare_choices(player_choice, computer_choice)


# let the player to player until they want to quit
player_want_to_play = True
while player_want_to_play:
    play_one_game(allowed_vals)
    # ask if the player wants to continue

