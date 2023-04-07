#!/usr/bin/python

import random
from interview import Interview

interview = Interview()

NUMBER_OF_ROUNDS = 1000

for round in range(NUMBER_OF_ROUNDS):
    for question in interview.get_next_question():
        print(question, end="")
        # This will be the agent.
        answer = random.choice(["yes", "no"])
        print(answer)
        interview.answer(answer)

    feedback, success, score = interview.get_test_feedback()
    print(interview.feedback_prompt, end="")
    print(feedback, end="")
    if success:
        print(f"Final score: {score}")
        break
    interview.start_next_round()
    print(interview.time_machine_use, end="")
