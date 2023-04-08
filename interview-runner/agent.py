import random

class Agent:
    """
    The agent class runs through multiple episodes until it either fails or solves the task.
    It collects the following data for performances.
    """
    def __init__(self):
        # TODO: Put in Alpaca or something similar here.
        # Note that the agent internal state is properly increasing all the time
        # due to the Decoder-only Transformer model, so it will only infer over
        # the new tokens received.
        # We clear the session by creating a new Agent.
        self.model = None
        # The list of messages so far.
        self.state = []

    def add_message(self, message):
        print(message, end="")
        self.state.append(message)

    def get_action(self):
        action = random.choice(["yes\n", "no\n"])
        self.add_message(action)
        return action

    def give_feedback(self, feedback):
        # The agent must store the change in its state when it processes feedback
        # so that we can use it to model agent internal state as it relates to
        # reinforcement learning.
        self.add_message(feedback)
        # TODO: Capture the change in the internal state over feedback.
    