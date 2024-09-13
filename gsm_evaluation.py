import json
import numpy as np
import time
import re
import argparse


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", default=3, type=int)
    parser.add_argument("--num_agents", default=3, type=int)

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


def answer_check(List, answer):
    if answer in List:
        return 1.0
    else:
        return 0.0


def compute_accuracy(gt, pred_solutions):
    answers = solve_math_problems(gt)

    if not answers:
        return None

    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if not pred_answer:
                pred_answer = solve_math_problems(pred_solution)

            pred_answers.append(pred_answer)

        return answer_check(pred_answers, answers)


if __name__ == "__main__":
    args = args_parse()
    num_agents = args.num_agents
    turns = args.turns
    file_name = f"{num_agents}agents_{turns}turns.json"

    with open(f"gsm_result_{file_name}", "r") as f:
        response_dict = json.load(f)

    questions = [response_dict[i]["question"] for i in range(len(response_dict))]

    performance = []

    for turn in range(turns):
        accuracies = []
        for idx in range(len(questions)):
            responses = [
                response_dict[idx]["agent_response"][i][turn] for i in range(num_agents)
            ]
            gt = response_dict[idx]["answer"]

            accurate = compute_accuracy(gt, responses)

            if accurate is not None:
                accuracies.append(float(accurate))
            else:
                accuracies.append(0.0)

        performance.append({f"{turn+1}_performance": np.mean(accuracies)})
        print({f"{turn+1}_performance": np.mean(accuracies)})

    print(f"The performance file 'gsm_performance_{file_name}' is saving...")
    with open(f"gsm_performance_{file_name}", "x") as f:
        json.dump(performance, f, indent=4)

    print("All done!!")
