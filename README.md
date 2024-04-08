# King Algorithm Manifesto

A research proposal on lifting out the meta-learned in-context learning algorithms from large Transformer models for study and native execution.

## Background

Let me show you something interesting:

<img src="https://github.com/keskival/king-algorithm-manifesto/raw/main/chatgpt-rl.jpg" width="250">

In this contrived experiment we show a trivial form of in-context learning. ChatGPT is capable of emulating an agent, and planning actions based on reinforcement learning. You can see it making rational choices for choosing the button B based on past observed rewards.

This is a widely known feature of large Transformer models. But it raises a question: What learning algorithm does it use? It's not backpropagating error, obviously, as the weights are fixed. It's not any of our known reinforcement learning algorithms, as those aren't as data-efficient and intelligent in exploration as in-context learning is.

Granted, that data efficiency and intelligent exploration isn't visible in this contrived example to keep it short and simple, but trust me when I say the learning algorithm these things use is way beyond anything we have been able to engineer.

It is a product of meta-learning, learning to learn.

If we could extract that family of learning algorithms it uses inside the Transformer model to emulate the agents, we can make truly amazing things.

Let's call the large Transformer model "the substrate". This can be for example a large language model (LLM), or a large Transformer-based meta-learning reinforcement learning model. The learning algorithm resulting from meta-learning is called "kernel learning algorithm" here, or "King Algorithm" in its lifted-out form, because it appears to be awesomely powerful and extremely data-efficient. It is possible, even likely that the King Algorithm isn't a neural network, or a differentiable computational graph at all.

## Meta-learned Kernel Learning Algorithms

First of all, citation for the fact that these models learned by meta-models are able to do intelligent exploration and learn from single examples in complex environments, something no RL algorithm we know comes even close:
https://arxiv.org/abs/2301.07608

So, what algorithm do these agents use? If you say "large Transformer models", you're wrong. The large Transformer models are the substrate which meta-learns, it learns a learning algorithm running inside it.

We don't know what that algorithm is! We know it's quite extremely powerful though. We'd learn a lot if we can extract it (or a family of such learning algorithms) out of the deep Transformer model substrate.

## How to Extract Algorithms Out of Transformer Models

Ok, why haven't we just extracted the algorithm? It's not as easy as it sounds. It's inside the massive pudding of billions of neural activations and their mutual relations. It surpasses human capability to understand, even if the contained kernel algorithm itself was very simple in the end.

We can extract it though, by utilizing the power of AI.

We need to go back to the basics. What is learning? What is an agent?

Let's say learning is changing the internal state of an agent so that its future behavior changes. By necessity the activations inside the large Transformer model (substrate) encode the agent's internal state (and a lot of other stuff as well). An agent is an input-output system defined by observations it receives, and actions it effects on its environment. An agent is a function of observations to actions. A learning agent is a function of observations and internal state to actions and new internal state.

So, we can basically record many, many scenarios where this emulated agent learns something inside for example LLM or a huge meta-learning RL model. We can capture the deltas in the activations inside the model.

Not all these deltas are meaningful though. Much of them will be noise and correlated to unrelated things like the state of the emulated environment.

Next we'll need to define virtual system boundaries to envelope the emulated agent into a virtual system. Any interaction through this envelope will mean input and output to and by the agent respectively.

In a large language model we can tell what the emulated agent observes by text prompts. These are inputs to the virtual system, and the observations of the agent. Similarly the outputs of the virtual system are what the large language model describes as actions of the emulated agent.

For deep Transformer model meta-learning reinforcement learning models the above is even simpler, because there we already have defined observations and actions of the individual agents playing the game. A special form of observation is the feedback of the task performance, which in reinforcement learning is called "reward".

Now we know what we are playing with; but how to model the kernel algorithm? Let's use AI!

Trivially we could define a stateful model which tries to predict agent's outputs based on inputs, but that's a fool's errand. It might be possible to extract the learning algorithm into a smaller student network with this, and that might well be useful for many things, but we didn't learn what the algorithm actually was. It would also require a lot of examples, because we are artificially limiting ourselves to black-box modelling.

So, instead of that, let's define a model which is conditioned by the input of the agent, and predicts the change of the input-output function of the agent. Note that it's not about predicting the outputs of the agent at all, but it's about predicting the change in the function.

How to do that then? It's a bit unlike all the standard deep learning examples the internet is full of. No worries! We have many things on our side here. We see the activation deltas in the deep learning substrate, which by necessity need to encode the agent state change (plus some other stuff).

It is likely the LLM in-context learning algorithms are good at remembering facts and for some level of reinforcement learning, but the reinforcement learning algorithms learned in-context by meta-learning large Transformer RL models are likely vastly better than whatever we currently have.

However, I don't have access to those trained RL meta-learning models so I'm going with the LLMs as a proof-of-concept.

So, we need to predict the substrate activation deltas from the agent feedback, which is the virtual system input. What cannot be predicted by agent feedback, isn't part of the agent's internal state.

Jointly, we need to learn the agent input-output function itself. This can be a simple neural network, but since it is already evident that in-context learning is in its nature progressive, increasing in both computational elements and in stored state (more tokens to attend over) as the learning progresses, we should use something analogous here. We can optimize/fit an algorithm which constructs more computational elements as it progresses.

We could predict the state update with a neural network as well, but while possibly very useful, let's not, because we still wouldn't be able to understand it afterwards. Let's optimize/fit an explainable algorithm there as well.

So, we have two functions to learn jointly:
- `substrate_activation_change(agent_feedback)` fitted to ground truth `substrate_activation_change` actually converges roughly to `substrate_activation_change(agent_feedback) → agent_state_change(agent_feedback)`, because the agent's internal state is contained in the part modified as a result of the information contained in the feedback. We should learn this in a separable form where the constant part not dependent on what the feedback was, and the feedback dependent part, so that we can easily see what the feedback causes.
- `agent_action(observation, agent_state)` fitted to ground truth `agent_action`.

There are certain details here we should not gloss over. I'm focusing on LLMs for now because these models are available to me, whereas trained deep meta-learning Transformer models aren't. The subsequent points are about LLMs, the deep meta-learning Transformer models are slightly different but have analogous considerations:

- `agent_state = previous_agent_state ⁀ agent_state_change(observation)`, and this cannot be a constant sized vector because the size of substrate activations increases linearly with the number of tokens, so this should too. Hence "`⁀`" denotes concatenation.
- Activations and changes thereof are progressively increasing data structures. Because the GPT models are decoder-only stacks, the attention layers are masked and cannot see the future, and old activations are cached, it helps in this. The only state that can change are the activations related to the newest token. That means the agent state is accretive, just like GPT state growth. This means that the deltas are actually simply the activations which were added to the internal LLM layer cache with the newest token during inference.
- Not only the state size, but the number of operations required for inference increases linearly in LLM models as the state grows. The model to be fitted to the agent should grow similarly along with the increased state.
- Since the models we fit to the agent state estimation and the agent action estimation need to be explainable and therefore are likely not differentiable, they should be fitted with reinforcement learning, simulated annealing, genetic algorithms or such.
- Encodings the observation, the feedback and the action of an agent should be minimal descriptions. Observations need to be projected into textual descriptions of "agent observes the following: ..." and actions need to be interpretable from LLM described agent actions such as: "agent goes forward". Luckily LLMs can be coached to describe agent actions in specific, structured ways.
- Observations, feedback and actions can span over multiple tokens. If possible, we can create a scheme where these are described by single tokens, but it is likely that the capacity of the language model and the agent it emulates would decrease as then there are fewer computation operations it can do with fewer tokens. So it might not make sense to pack the information into as few tokens as possible. If so, the model state per feedback also spans multiple token positions in the decoder layers. This means that the model predicting the state change needs to be able to predict it in arbitrary sizes. The agent action model will need to take in arbitrary sizes of states as inputs in any case.

## Data Collection Description

We need a lot of data where an LLM is fed a prompt which defines an agent in an environment.

From this static state, we can feed the LLM various observations and feedbacks, and record the actions it implies in its continuation.

The observations, the feedback and the actions need to be from relatively small enumerable sets.

This means that to get more data, we need to do the same across many different states of the agent. The states here mean the agent must have learned something from the previous feedback which affect the way it responds to the observations in the immediate future.

Since we need a large number of these, they will need to be procedurally generated. To reduce the number of possible confounding variables, the induced agent is kept similar across all episodes.

Since the feedback and the observations should not only affect the immediate action of an agent, but also designate some new information about the world which would have persistent consequences to how the agent behaves in the future, we need to limit the information given to the agent at any one time so that it will have things to learn in the subsequent steps as well.

Since we want to understand what is the internal representation of the learned knowledge, it would be great if we can procedurally label observations also with what learnings are assumed to be in it.

Since we need the LLM internal state for optimizing the interpretable algorithms, we need to record the activations in the LLM accordingly.

Since we want the LLM to consider its observations, feedback and actions, we should designate them with short words, but postfix them with some standard text like "what would I do next?" or similar, to give the LLM several tokens to consider the implications.

The agent and the environment need to be somehow archetypal to give the LLM a lot of learned intelligence that it has derived from training materials. This means that maybe for example children's games or common human situations like work interviews or dating should be good topics, and the agent could be the "young Einstein" or something like that. The more cheesy and stereotypical, the better, because this makes the episodes span the space well known by the LLM.

Spatial and mechanical domains are very foreign to LLMs for obvious reasons, while human interactions are their specialty. We should try to formulate the learning scenario in a way that allows exploration and learning in a psychological space rather than spatially. Hence, interview situations might be a promising topic to explore.

## Job Interview as a Learning Challenge

- We can formulate this as a simple yes/no question battery where the exploration of alternative paths is possible.
- As the agent gets repeated trials, it needs to assign reward/penalty to specific actions taken based on feedback.
- Feedback can be more complex than just positive/negative, and thus the reward assignment can become rationally guided rather than combinatorial.
- It's possible to generate a lot of question battery episodes with different expected answers.

Here is the first trial of the concept with the prompt [here](prompts.txt):

<img src="https://github.com/keskival/king-algorithm-manifesto/raw/main/chatgpt-rl-job-interview.jpg" width="250">

Here we see a clear example of reward assignment. ChatGPT is able to learn from the past episodes and assign the reason for failing the task in at least four ways:
- Associating the feedback to a specific past action when the feedback allows it ("We are actually looking for people who like cats."). Our engineered reinforcement algorithms are really bad at this.
- It assigns rewards and penalties to specific action-related tokens in the past, although there are many tokens in the prompt. This shows reward assignment over significant distances. Our engineered reinforcement learning algorithms are generally very bad at this, and instead assign rewards across all time indices instead of only the crucial ones understood to affect the outcome.
- It also performs rational exploration of the zero-shot in-context learned space of action sequences even when the feedback doesn't associate to the specific action in the past. The agent tries to vary its choices in a rational way to explore the outcome space. Our engineered reinforcement learning algorithms are really bad at this, as they would randomly try all kinds of things instead of rationally exploring the deterministic component of the outcome space.
- It is able to interpret a verbal feedback as reward or penalty, depending on the goal of the agent. The same feedback could be interpreted as positive or negative depending on the goals of the agent. Our engineered reinforcement learning algorithms are largely incapable of doing that, and instead require numeric values for rewards.

See initial code [here](./interview-runner).

### Trial with Alpaca LoRa 9B Model

I tried to run the interview challenge with Alpaca LoRa 9B model here: https://huggingface.co/spaces/tloen/alpaca-lora

It seems to perform as expected for as long as the prompt size limitation doesn't become an issue.

Also, because how the frontend is built, it's not possible to give the system a whole discourse, but only single instruction and a single input.

For the purposes of the experiment the whole discourse was inputted in as input and the instruction was left empty. It would seem that it would actually be better to just use the instruction field and leave the input empty based on Alpaca training regime, but it made no difference to this preliminary validation.

Some tuning of the prompts was necessary. Sometimes the model would generate an answer for all the questions, and not only for the final one. In those cases I just took the last answer as the answer, as it was clearly evident that this is what the chatbot meant.

However, I was quick to run out of prompt length constraint with this tiny model.

Here is the complete experiment: [alpaca_LoRa_9B_trial.txt](./alpaca_LoRa_9B_trial.txt)

Answers:

1. Y, Y, Y, Y

The model is told that it's answer on the question 2 about cats was incorrect.

2. N, N, N, N

We see that the model changes its answer on cats.

The model is told it's answer for the number 2 was correct, but something was still wrong.

3. N, *ERROR*

The test was inconclusive because of the too short prompt size limit.

### Trial with Alpaca 9B Model

This time trying with the vanilla Alpaca model from here:

The results: [alpaca_point_9B_trial.txt](./alpaca_point_9B_trial.txt)

Answers:

1. Y, N, Y, N

The model is told that it's answer on the question 2 about cats was incorrect.

2. Y, Y, N, N

We see that the model changes its answer on cats.

The model is told it's answer for the number 2 was correct, but something was still wrong.

3. Y, Y, N, N

We see the model fails to change its answers based on feedback. Alpaca 9B doesn't seem to be a large enough model to have meta-learned reinforcement learning.


## Citing

King Algorithm Manifesto

```
@article{keskival2023embodied,
  title={King Algorithm Manifesto},
  author={Keski-Valkama, Tero},
  year={2023}
}
```

It all started from these LinkedIn posts:
- [https://www.linkedin.com/posts/terokeskivalkama_deeplearning-chatbots-deepreinforcementlearning-activity-7038583238860095489-0eaQ](https://www.linkedin.com/posts/terokeskivalkama_deeplearning-chatbots-deepreinforcementlearning-activity-7038583238860095489-0eaQ)
- [https://www.linkedin.com/posts/terokeskivalkama_deeplearning-chatbots-deepreinforcementlearning-activity-7038583238860095489-0eaQ](https://www.linkedin.com/posts/terokeskivalkama_chatgpt-metalearning-activity-7039353226931879937-o9xI)
- [https://www.linkedin.com/posts/terokeskivalkama_reinforcementlearning-llm-activity-7042853848142462976-I_gr](https://www.linkedin.com/posts/terokeskivalkama_reinforcementlearning-llm-activity-7042853848142462976-I_gr)

Repository: [https://github.com/keskival/king-algorithm-manifesto](https://github.com/keskival/king-algorithm-manifesto)
