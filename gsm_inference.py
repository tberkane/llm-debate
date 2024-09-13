import requests
import openai
import json
import numpy as np
import random
import time
from tqdm import tqdm
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", default=2, type=int)
    parser.add_argument("--num_agents", default=2, type=int)

    return parser.parse_args()


def load_json(prompt_path):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)
    return prompt_dict


def construct_message(agent_context, instruction):
    prefix_string = "Here are a list of opinions from different agents: "

    prefix_string = (
        prefix_string
        + agent_context
        + "\n\n Write a summary of the different opinions from each of the individual agent."
    )

    message = [{"role": "user", "content": prefix_string}]

    completion = pipe(message, **generation_args)

    prefix_string = f"Here is a summary of responses from other agents: {completion}"
    prefix_string = (
        prefix_string
        + "\n\n Use this summarization carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response."
        + instruction
    )
    return prefix_string


def summarize_message(agent_contexts, instruction, idx):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = (
        prefix_string
        + "\n\n Write a summary of the different opinions from each of the individual agent."
    )
    completion = construct_message(prefix_string, instruction, idx)

    return completion


def generate_gsm(agents, question):
    agent_contexts = [
        [
            {
                "model": f"model_{agent_id}",
                "content": f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.",
            }
        ]
        for agent_id in range(agents)
    ]
    return agent_contexts


def read_jsonl(path: str):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh.readlines() if line]


torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}


if __name__ == "__main__":
    args = args_parse()

    def generate_answer(question):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.",
            },
        ]
        output = pipe(messages, **generation_args)

        return output[0]["generated_text"]

    def prompt_formatting(agent_id, instruction):
        prompt = f"User:\n{instruction}\n\nAssistant:\n"
        return {
            "model": f"model_{agent_id}",
            "content": prompt,
        }

    num_agents = args.num_agents
    turns = args.turns
    random.seed(0)

    evaluation = 100

    generated_description = []

    questions = read_jsonl("gsm8k_test.jsonl")
    random.shuffle(questions)

    for idx in tqdm(range(evaluation)):
        question = questions[idx]["question"]
        answer = questions[idx]["answer"]

        agent_contexts = generate_gsm(num_agents, question)

        print(f"# Question No.{idx+1} starts...")

        message = []

        # Debate
        for debate in range(turns + 1):
            # Refer to the summarized previous response
            if debate != 0:
                message.append(
                    summarize_message(agent_contexts, question, 2 * debate - 1)
                )
                for i in range(len(agent_contexts)):
                    agent_contexts[i].append(
                        prompt_formatting(agent_contexts[i][-1]["model"], message)
                    )

            for agent_context in agent_contexts:
                # Generate new response based on summarized response
                completion = generate_answer(agent_context[-1]["content"])
                agent_context.append(completion)

        print(f"# Question No.{idx+1} debate is ended.")

        models_response = {
            f"model_{agent_id}": [
                agent_contexts[agent_id][1]["content"],
                agent_contexts[agent_id][3]["content"],
                agent_contexts[agent_id][-1]["content"],
            ]
            for agent_id in range(num_agents)
        }
        response_summarization = [message[0], message[1]]
        generated_description.append(
            {
                "question_id": idx,
                "question": question,
                "agent_response": models_response,
                "summarization": response_summarization,
                "answer": answer,
            }
        )

    file_name = f"{num_agents}agents_{turns}turns.json"

    print(f"The result file 'gsm_result_{file_name}' is saving...")
    with open(f"gsm_result_{file_name}", "x") as f:
        json.dump(generated_description, f, indent=4)

    print("All done!!")
