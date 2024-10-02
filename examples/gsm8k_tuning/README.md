## Math Agent distillation
This example demonstrates how to make small LLama model better at math. We will use [GSM8k](https://huggingface.co/datasets/openai/gsm8k). dataset. It is a collection of 8,000 math word problems with their corresponding equations.

Steps:
- First we build a basic math agent that uses LLama 3.1 70B model and calculator tool to solve the math word problems: [math_agent.py](math_agent.py).
- Then we [run it as a teacher](produce_teacher_tapes.py) and collect the tapes of the successful solutions and produce training data for the Math Agent from them. How to run: `python produce_teacher_tapes.py`
- After that, we [fine-tune smaller LLama 3.1 8B model](finetune_student.py) on the training data to get a tuned Math Agent. How to run: `python finetune_student.py`
- Finally, we [evaluate the tuned Math Agent](evaluate_student.py) on the GSM8K test set, comparing accuracy of the teacher agent, student agent before tuning, and student agent after tuning. How to run: `python evaluate_student.py`

RL tuning on both succesfull and unsuccesfull solutions is coming soon. Stay tuned!