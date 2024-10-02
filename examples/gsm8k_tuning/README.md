## Math Agent distillation
This example demonstrates how to improve the math skills of a small LLama model. We will use the [GSM8k](https://huggingface.co/datasets/openai/gsm8k) dataset, which consists of 8,000 math word problems with their corresponding equations.

Steps:
First, we build a basic math agent that uses the LLama 3.1 70B model and calculator tool to solve math word problems: [math_agent.py](math_agent.py).
- Then we [run it as a teacher](produce_teacher_tapes.py), collect the tapes of the successful solutions, and produce training data for the Math Agent from them. How to run: `python produce_teacher_tapes.py`
- After that, we [fine-tune smaller LLama 3.1 8B model](finetune_student.py) on the training data to get a tuned Math Agent. How to run: `python finetune_student.py`
- Finally, we [evaluate the tuned Math Agent](evaluate_student.py) on the GSM8K test set, comparing the accuracy of the teacher agent, student agent before tuning, and student agent after tuning. How to run: `python evaluate_student.py`

<img width="526" alt="image" src="https://github.com/user-attachments/assets/a7aa2908-2a86-4b85-92d2-8c133e9ac0ff">

RL tuning on both successful and unsuccessful solutions is coming soon. Stay tuned!
