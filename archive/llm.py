from gpt4all import GPT4All
import subprocess

# Step 1: Prompt the computer vision module to extract objects from an image

# Step 2: use the output from the computer vision module for LLM to translate to FOL language

# Step 3: Prompt the planner module to generate a plan based on the FOL statements

# Step 4: Use the LLM to translate the plan to natural language solutions


model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf") # downloads / loads a 4.66GB LLM
with model.chat_session():
    print(model.generate("How can I run LLMs efficiently on my laptop?", max_tokens=1024))