#!/usr/bin/python

from interview import Interview
from agent import Agent


NUMBER_OF_ROUNDS = 1000
MAX_NUMBER_OF_EPISODES = 1000

for round in range(NUMBER_OF_ROUNDS):
    # Each round starts from empty state, new agent and a new interview process.
    agent = Agent()
    interview = Interview()
    for episode in range(MAX_NUMBER_OF_EPISODES):
        for question in interview.get_next_question():
            agent.add_message(question)
            # This will be the agent.
            answer = agent.get_action()
            interview.answer(answer)

        feedback, success, score = interview.get_test_feedback()
        agent.add_message(interview.feedback_prompt)
        state_delta = agent.give_feedback(feedback)
        if success:
            print(f"Final score: {score}")
            break
        interview.start_next_round()
        agent.add_message(interview.time_machine_use)
