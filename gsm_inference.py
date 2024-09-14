import argparse
import json
import random
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default=2, type=int)
    parser.add_argument("--num-agents", default=2, type=int)
    parser.add_argument("--evaluation", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    return parser.parse_args()


def setup_model_and_pipeline():
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def format_prompt(question, instruction, previous_response=None, summary=None):
    prompt = f"""Question: {question}

    Your task is to solve this math problem. Please follow these guidelines:
    1. Provide a final answer in the format: \\boxed{{your_numerical_answer_here}}
    2. Ensure that the boxed answer is the last thing in your response.

    {f'Previous response: {previous_response}' if previous_response else ''}

    {f'Summary of other agents thoughts: {summary}' if summary else ''}

    {instruction}

    Your response:
    """
    return prompt


def generate_text(pipe, input_text, generation_args, debug=False):
    message = [{"role": "user", "content": input_text}]
    if debug:
        print(f"\n[DEBUG] Input to model:\n{input_text}")
    output = pipe(message, **generation_args)
    generated_text = output[0]["generated_text"]
    if debug:
        print(f"[DEBUG] Model output:\n{generated_text}")
    return generated_text


def summarize_responses(pipe, agent_contexts, question, generation_args, debug=False):
    summary_prompt = (
        "Here are the responses from different agents to the following question:\n\n"
    )
    summary_prompt += f"Question: {question}\n\n"
    for idx, agent in enumerate(agent_contexts):
        summary_prompt += f"Agent {idx} response:\n{agent[-1]['content']}\n\n"
    summary_prompt += "Please provide a concise summary of the different approaches and answers given by the agents."

    if debug:
        print(f"\n[DEBUG] Summarization prompt:\n{summary_prompt}")

    summary = generate_text(pipe, summary_prompt, generation_args, debug)

    if debug:
        print(f"[DEBUG] Summary:\n{summary}")

    return summary


def generate_gsm(num_agents, question):
    instruction = "Please solve the problem."
    return [
        [{"model": f"model_{i}", "content": format_prompt(question, instruction)}]
        for i in range(num_agents)
    ]


def read_jsonl(path):
    with open(path, "r") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_debate(pipe, question, num_agents, num_rounds, generation_args, debug=False):
    agent_contexts = generate_gsm(num_agents, question)
    summaries = []

    if debug:
        print(f"\n[DEBUG] Starting debate on question:\n{question}")

    for round in range(num_rounds + 1):
        if debug:
            print(f"\n[DEBUG] Round {round} of debate")

        if round != 0:
            summary = summarize_responses(
                pipe, agent_contexts, question, generation_args, debug
            )
            summaries.append(summary)
            instruction = "Based on the summary of other agents' thoughts, please reconsider your approach. If you find merit in other methods, incorporate them. If you believe your method is correct, defend it with additional explanation."
            for agent_context in agent_contexts:
                prompt = format_prompt(
                    question, instruction, agent_context[-1]["content"], summary
                )
                agent_context.append(
                    {"model": agent_context[-1]["model"], "content": prompt}
                )
        else:
            instruction = (
                "Please solve the problem and explain your reasoning step by step."
            )

        for i, agent_context in enumerate(agent_contexts):
            if debug:
                print(f"\n[DEBUG] Agent {i} thinking...")
            response = generate_text(
                pipe, agent_context[-1]["content"], generation_args, debug
            )
            agent_context.append(
                {"model": agent_context[-1]["model"], "content": response}
            )

    return agent_contexts, summaries


def main():
    args = parse_arguments()
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)

    pipe = setup_model_and_pipeline()
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": args.temperature,
    }

    questions = read_jsonl("gsm8k_test.jsonl")
    random.shuffle(questions)

    results = []
    file_name = f"gsm_result_{args.num_agents}agents_{args.rounds}turns_{args.evaluation}eval.json"

    for idx in tqdm(range(args.evaluation)):
        if args.debug:
            print(f"\n[DEBUG] Starting evaluation {idx + 1}/{args.evaluation}")

        question = questions[idx]["question"]
        answer = questions[idx]["answer"]

        agent_contexts, summaries = run_debate(
            pipe, question, args.num_agents, args.rounds, generation_args, args.debug
        )

        models_response = {
            f"model_{i}": [
                context[1]["content"],
                context[3]["content"],
                context[-1]["content"],
            ]
            for i, context in enumerate(agent_contexts)
        }

        results.append(
            {
                "question_id": idx,
                "question": question,
                "agent_response": models_response,
                "summarization": summaries,
                "answer": answer,
            }
        )

        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)

        if args.debug:
            print(f"[DEBUG] Results saved after question {idx + 1}")

    print(f"Results saved to '{file_name}'")
    print("All done!")


if __name__ == "__main__":
    main()
