#!/usr/bin/python

from interview import Interview

interview = Interview()

for question in interview.get_next_question():
    print(question)
    answer = "yes"
    print(answer)
    interview.answer(answer)

print(interview.feedback_prompt + interview.get_test_feedback())
