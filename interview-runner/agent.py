import random

class Agent:
    """
    The agent class runs through multiple episodes until it either fails or solves the task.
    It collects the following data for performances.

    We need to store for every decision taken:
    - (feedback states, observation, observation state, decision, decision state) for every decision taken. The last feedback is the neural activation of the last
      decoder column between the start of feedback and end of feedback.
      The observation is the categorical id of the question.
      The observation state is the neural activation columns between the start and end of one question.
      Decision is a boolean yes/no which corresponds to the answers. Decision state is the column of activations for the yes/no answer.
    For every feedback, that is, episode:
    - (feedback, feedback state) for every feedback received. The feedback is a categorical id of the feedback given.
    - We could store all the states between other prompts as well, as they represent thinking about inputs already seen before,
      but I don't think those are useful for modelling the agent learning algorithm.
    For every round:
    - (score) for every round.
    These tuples can be used to fit different models which tell us how these system learn.
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
        # TODO: Capture both the feedback and the change in the internal state over feedback.
