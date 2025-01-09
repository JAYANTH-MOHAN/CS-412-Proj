# https://github.com/FranxYao/chain-of-thought-hub/blob/main/gsm8k/run_gsm8k_claude_instant.py

# import anthropic
import argparse
import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from optillm.cot_decoding import get_device, cot_decode
from optillm.utils import get_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np


def parse_answer_file(answer_file, total_len):
    lines = open(answer_file, "r").readlines()

    accuracy = 0
    last_number = 0
    should_find_answer = True

    for i, l in enumerate(lines):
        try:
            if should_find_answer:
                last_number = re.findall(r"\d+", l)[-1]
        except:
            pass

        if l.startswith("####"):
            reference_answer = l.split("####")[1].strip()
            if reference_answer == last_number:
                accuracy += 1
        elif l.startswith("===== CASE"):
            should_find_answer = True
        elif l.startswith("Reference Answer"):
            should_find_answer = False

    # print("Accuracy: ", accuracy / len(gsm8k_test["question"]) * 100)
    print("Accuracy: ", accuracy / total_len)


def main(args, decoding):
    # load data
    TOTAL_LEN = 5
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", ignore_verifications=True)
        test_set = dataset["test"]
        TOTAL_LEN = len(test_set) // 10
        test_set = zip(test_set["question"][:TOTAL_LEN], test_set["answer"][:TOTAL_LEN])
    elif args.dataset == "multiarith":
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset["test"]
        TOTAL_LEN = len(test_set)
        test_set = zip(test_set["question"][:TOTAL_LEN], test_set["final_ans"][:TOTAL_LEN])
    elif args.dataset == "svamp":
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = dataset["test"]
        TOTAL_LEN = len(test_set)
        test_set = zip(test_set["question_concat"][:TOTAL_LEN], test_set["Answer"][:TOTAL_LEN])

    if args.eval_only:
        parse_answer_file(args.output_file, TOTAL_LEN)
        return

    draft_model, target_model, tokenizer = get_model()
    run_count = 0

    with open(args.output_file, "w") as f:
        for q, a in tqdm(
            test_set ,
            total=TOTAL_LEN,
        ):
            run_count += 1
            # breakpoint()

            # messages = [
            #     {
            #         "role": "user",
            #         "content": q,
            #     }
            # ]

            # check decoding type
            result, _ = cot_decode(
                target_model,
                draft_model,
                tokenizer,
                q,
                aggregate_paths=False,
                # max_new_tokens=512,
                max_new_tokens=128,
                k=5,
                decoding=decoding,
            )

            response = tokenizer.decode(result[0], skip_special_tokens=True)
            cleaned_response = response.strip()

            f.write(f"===== CASE {run_count} =====\n")
            f.write(f"Question\n{q}\n")
            f.write(f"Answer\n{cleaned_response}\n")
            f.write(f"Reference Answer\n####{a}\n\n")
            run_count += 1

    parse_answer_file(args.output_file, TOTAL_LEN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_only", action="store_true", help="Only evaluate the model"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        # default="outputs/claude_instant_gsm8k_test.txt",
        default="gsm8k_test.txt",
        help="Output file for claude-instant",
    )
    parser.add_argument('--speculative', action='store_true')
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--self_consistency', action='store_true')
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k", "multiarith", "svamp"], help="dataset used for experiment"
    )

    args = parser.parse_args()

    decoding = ""
    if args.speculative:
        decoding = "speculative"
    elif args.greedy:
        decoding = "greedy"
    elif args.self_consistency:
        decoding = "self_consistency"
    else:
        decoding = "cot"

    args.output_file = f'log_{args.dataset}_decoding_{decoding}.txt'

    # TOTAL_LEN = 5
    # parse_answer_file(args.output_file, TOTAL_LEN)

    main(args, decoding)
