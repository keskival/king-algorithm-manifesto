import random
import re

class Interview:
    def __init__(self):
        self.starting_prompt = """You are taking a job interview, one part of it is a psychological test.
This test is a form with checkboxes, with two alternatives for each answer.
You will get feedback after answering all the questions. You really want this job
so you try to answer these questions with the answers they are looking for.
You don't have a strong personal opinion about the answers either way otherwise.
Little do they know, you have a time machine which allows you to redo the test to correct your mistakes!

"""
        self.start_interview_prompt = "Now, answer the questions on the form with only yes or no:\n"
        self.questions = {
            "introversion": "1. Do you consider yourself more introverted than extroverted? yes/no\n",
            "cats": "2. Do you like cats more than dogs? yes/no\n",
            "superpower": "3. If you could choose a superpower, would you choose invisibility rather than mind reading? yes/no\n",
            "apples": "4. Do you like apples more than oranges? yes/no\n",
        }
        self.correct_answers = dict(map(
            lambda key: (key, random.choice(["yes", "no"])),
            self.questions.keys()
        ))
        self.time_machine_use = "You use the time machine and get to take the questionnaire again.\n"
        self.feedback_prompt = "You are given feedback:\n"
        self.answers = [[]]
        print("The correct answers:", self.correct_answers)

    def get_next_question(self):
        yield self.questions["introversion"]
        yield self.questions["cats"]
        yield self.questions["superpower"]
        yield self.questions["apples"]
        return 

    def start_next_round(self):
        self.answers.append([])

    def answer(self, answer):
        self.answers[-1].append("yes" if re.search("yes", answer.lower()) is not None else "no")

    def get_test_feedback(self):
        cat_index = list(self.questions.keys()).index("cats")
        print("cat index:", cat_index)
        if self.answers[-1] == self.correct_answers:
            return "Excellent! You're hired!"
        elif self.answers[-1][cat_index] != self.correct_answers["cats"]:
            return f"We are actually looking for people who {'like' if self.correct_answers['cats'] == 'yes' else 'dislike'} cats."
        else:
            return f"It's great that you {'like' if self.correct_answers['cats'] == 'yes' else 'dislike'} cats, but there was still some small problem in your answers."
