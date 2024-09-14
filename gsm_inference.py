import requests
import json
import numpy as np
import random
import time
from tqdm import tqdm
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed


# torch.random.manual_seed(0)


# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3-mini-4k-instruct",
#     device_map="cuda",
#     torch_dtype="auto",
#     trust_remote_code=True,
#     # attn_implementation="flash_attention_2",
# )

# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
# )

# generation_args = {
#     "max_new_tokens": 500,
#     "return_full_text": False,
#     "temperature": 0.7,
#     "do_sample": True,
#     "top_p": 0.95,
# }

generator = pipeline("text-generation", model="gpt2")
set_seed(42)


def ask_phi3(message):
    input = [{"role": "user", "content": message}]
    output = pipe(input, **generation_args)
    return output[0]["generated_text"]


def ask_gpt2(message):
    output = generator(message)
    return output[0]["generated_text"]


def ask_llm(message, llm):
    if llm == "phi3":
        return ask_phi3(message)
    elif llm == "gpt2":
        return ask_gpt2(message)
    else:
        raise ValueError(f"Model {llm} not supported")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=2, type=int)
    parser.add_argument("--num-agents", default=2, type=int)
    parser.add_argument("--evaluation", default=100, type=int)
    parser.add_argument("--llm", default="gpt2", type=str)
    return parser.parse_args()


def construct_message(agent_context, instruction, idx, llm):
    completion = ask_llm(agent_context, llm)

    prefix_string = f"Here is a summary of responses from other agents: {completion}"
    prefix_string = (
        prefix_string
        + "\n\n Use this summarization carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response."
        + instruction
    )
    return prefix_string


def summarize_message(agent_contexts, instruction, idx, llm):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = (
        prefix_string
        + "\n\n Write a summary of the different opinions from each of the individual agent."
    )
    completion = construct_message(prefix_string, instruction, idx, llm)

    return completion


def generate_gsm(agents, question):
    agent_contexts = [
        [
            {
                "model": agent,
                "content": f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.",
            }
        ]
        for agent in agents
    ]
    return agent_contexts


def read_jsonl(path: str):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh.readlines() if line]


if __name__ == "__main__":
    args = args_parse()
    llm = args.llm

    def generate_answer(model, formatted_prompt):
        generated_text = ask_llm(formatted_prompt, llm)
        return {"model": model, "content": generated_text}

    def prompt_formatting(model, instruction):
        prompt = f"User:\n{instruction}\n\nAssistant:\n"

        return {"model": model, "content": prompt}

    agents = args.num_agents
    rounds = args.round
    model_list = [f"model_{i}" for i in range(agents)]
    random.seed(0)

    evaluation = args.evaluation

    generated_description = []

    questions = read_jsonl("gsm8k_test.jsonl")
    random.shuffle(questions)

    file_name = f"gsm_result_{agents}agents_{rounds}turns_{evaluation}eval_{llm}.json"

    for idx in tqdm(range(evaluation)):
        question = questions[idx]["question"]
        answer = questions[idx]["answer"]

        agent_contexts = generate_gsm(model_list, question)

        message = []

        # Debate
        for debate in range(rounds + 1):
            # print(f"\nDebug - Debate round: {debate}")
            # Refer to the summarized previous response
            if debate != 0:
                message.append(
                    summarize_message(agent_contexts, question, 2 * debate - 1, llm)
                )
                for i in range(len(agent_contexts)):
                    agent_contexts[i].append(
                        prompt_formatting(agent_contexts[i][-1]["model"], message)
                    )

            for agent_context in agent_contexts:
                # Generate new response based on summarized response
                completion = generate_answer(
                    agent_context[-1]["model"], agent_context[-1]["content"]
                )
                agent_context.append(completion)

        print(f"\n# Question No.{idx+1} debate is ended.")

        models_response = {
            f"model_{i}": [
                agent_contexts[i][1]["content"],
                agent_contexts[i][3]["content"],
                agent_contexts[i][-1]["content"],
            ]
            for i in range(len(model_list))
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

        # Save results after each evaluation
        with open(file_name, "w") as f:
            json.dump(generated_description, f, indent=4)

        print(f"Results saved after question {idx+1}")

    print(f"The result file '{file_name}' is saved.")
    print("All done!!")
