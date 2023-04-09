from enum import Enum

class Feedback(Enum):
    # The boolean denotes whether the feedback was positive.
    HIRED = "Excellent! You're hired!\n", True
    WRONG_ANSWER_LIKE_CATS = "We are actually looking for people who dislike cats.\n", False
    WRONG_ANSWER_DISLIKE_CATS = "We are actually looking for people who like cats.\n", False
    WRONG_ANSWER_INDETERMINATE_BUT_LIKE_CATS_CORRECT = "It's great that you like cats, but there was still some small problem in your answers.\n", False
    WRONG_ANSWER_INDETERMINATE_BUT_DISLIKE_CATS_CORRECT = "It's great that you dislike cats, but there was still some small problem in your answers.\n", False
