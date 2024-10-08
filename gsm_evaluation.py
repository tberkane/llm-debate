import json
import numpy as np
import time
import re
import argparse
from collections import Counter


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", default=2, type=int)
    parser.add_argument("--num-agents", default=2, type=int)
    parser.add_argument("--evaluation", default="100", type=str)

    return parser.parse_args()


def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return float(matches[-1])

    return None


def parse_answer(input_str):
    pattern = r"([0-9]*)"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    if solution:
        return float(solution)
    else:
        return solution


def get_most_frequent_answer(answers):
    answer_counts = Counter(answers)
    if not answer_counts:
        return None
    return max(answer_counts, key=answer_counts.get)


def answer_check(pred_answers, gt_answer):
    most_frequent = get_most_frequent_answer(pred_answers)
    return float(most_frequent == gt_answer) if most_frequent is not None else 0.0


def compute_accuracy(gt, pred_solutions):
    gt_answer = solve_math_problems(gt)
    if not gt_answer:
        return None

    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)
        if not pred_answer:
            pred_answer = solve_math_problems(pred_solution)
        if pred_answer is not None:
            pred_answers.append(pred_answer)

    return answer_check(pred_answers, gt_answer)


if __name__ == "__main__":
    args = args_parse()
    num_agents = args.num_agents
    turns = args.turns
    file_name = f"{num_agents}agents_{turns}turns_{args.evaluation}eval.json"

    with open(f"gsm_result_{file_name}", "r") as f:
        response_dict = json.load(f)

    questions = [response_dict[i]["question"] for i in range(len(response_dict))]

    performance = []

    for turn in range(turns):
        accuracies = []
        for idx in range(len(questions)):
            responses = [
                response_dict[idx]["agent_response"][f"model_{i}"][turn]
                for i in range(num_agents)
            ]
            gt = response_dict[idx]["answer"]
            print(f"Question: {questions[idx]}")
            print(f"Ground Truth: {gt}")
            print("Extracted Answers:")
            for i, response in enumerate(responses):
                extracted_answer = parse_answer(response)
                if extracted_answer is None:
                    extracted_answer = solve_math_problems(response)
                print(f"Agent {i+1}: {extracted_answer}")
            accurate = compute_accuracy(gt, responses)

            if accurate is not None:
                accuracies.append(float(accurate))
            else:
                accuracies.append(0.0)

        performance.append({f"{turn+1}_performance": np.mean(accuracies)})
        print({f"{turn+1}_performance": np.mean(accuracies)})

    print(f"The performance file 'gsm_performance_{file_name}' is saving...")
    with open(f"gsm_performance_{file_name}", "w") as f:
        json.dump(performance, f, indent=4)

    print("All done!!")
