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


def generate_text(pipe, input_text, generation_args):
    message = [{"role": "user", "content": input_text}]
    output = pipe(message, **generation_args)
    return output[0]["generated_text"]


def summarize_responses(pipe, agent_contexts, instruction, generation_args):
    summary_prompt = "Here are a list of opinions from different agents: "
    for idx, agent in enumerate(agent_contexts):
        summary_prompt += f"\n\nResponse from agent {idx}: ```{agent[-1]['content']}```"
    summary_prompt += "\n\nWrite a summary of the different opinions from each of the individual agent."

    summary = generate_text(pipe, summary_prompt, generation_args)
    return f"Here is a summary of responses from other agents: {summary}\n\nUse this summarization carefully as additional advice. Can you provide an updated answer? Make sure to state your answer at the end of the response.{instruction}"


def generate_gsm(num_agents, question):
    return [
        [
            {
                "model": f"model_{i}",
                "content": f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, and should be the very last numerical value mentioned in your response.",
            }
        ]
        for i in range(num_agents)
    ]


def read_jsonl(path):
    with open(path, "r") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_debate(pipe, question, num_agents, num_rounds, generation_args):
    agent_contexts = generate_gsm(num_agents, question)
    summaries = []

    for round in range(num_rounds + 1):
        if round != 0:
            summary = summarize_responses(
                pipe, agent_contexts, question, generation_args
            )
            summaries.append(summary)
            for agent_context in agent_contexts:
                agent_context.append(
                    {"model": agent_context[-1]["model"], "content": summary}
                )

        for agent_context in agent_contexts:
            response = generate_text(
                pipe, agent_context[-1]["content"], generation_args
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
        question = questions[idx]["question"]
        answer = questions[idx]["answer"]

        agent_contexts, summaries = run_debate(
            pipe, question, args.num_agents, args.rounds, generation_args
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

    print(f"Results saved to '{file_name}'")
    print("All done!")


if __name__ == "__main__":
    main()
