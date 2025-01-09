import contexttimer
from optillm.speculative_decoding import contrastive_sampling, speculative_sampling
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np
import re
from optillm.utils import get_model, truncate


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def calculate_confidence(
    logits: List[torch.Tensor], answer_ids: torch.Tensor, is_logits=True
) -> float:
    """
    Calculate the confidence score (Δ) as specified in the paper.

    Args:
        logits: List of logits for each decoding step
        answer_ids: Tensor of token ids for the answer

    Returns:
        Confidence score (Δ)
    """
    confidence_sum = 0.0
    valid_tokens = 0
    for t, token_id in enumerate(answer_ids):
        if t >= len(logits):
            break
        token_logits = logits[t]
        if is_logits:
            probs = torch.softmax(token_logits, dim=-1).squeeze()
        else:
            probs = token_logits
        if probs.size(-1) > 1:
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[0] - top_2_probs[1]).item()
            else:
                confidence_sum += 1.0  # Max confidence if there's only one token
        else:
            confidence_sum += 1.0  # Max confidence if there's only one token
        valid_tokens += 1
    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0


def aggregate_paths_based_on_scores(
    paths: List[Tuple[str, float]]
) -> Tuple[str, float]:
    """Aggregate multiple paths based on their confidence scores."""
    answer_scores = {}
    for answer, delta in paths:
        answer_scores[answer] = answer_scores.get(answer, 0) + delta
    best_answer = max(answer_scores, key=answer_scores.get)
    return best_answer, answer_scores[best_answer]


def get_final_ans_span(ans_ids, ans_text, device, tokenizer):
    numerical_ans = re.findall(r"\d+", ans_text)[-1]
    numerical_ans_ids = (
        tokenizer.encode(numerical_ans, return_tensors="pt").to(device).squeeze()
    )[
        1:
    ]  # remove bos
    whole_len = ans_ids.size(-1)
    num_len = numerical_ans_ids.size(-1)
    for i in range(whole_len - numerical_ans_ids.size(-1), -1, -1):
        if torch.all(torch.eq(ans_ids[i : i + num_len], numerical_ans_ids)):
            return i, i + num_len


# https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/main.py
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 1
    profile_filename = f"./profile_logs/{print_prefix}"

    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=1, active=2, repeat=1, skip_first=0
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profile_filename
                ),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME):
                    _, total_len = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME):
                _, total_len = fn(*args, **kwargs)

    print(
        f"\n [benchmark] {print_prefix}, tokens/sec: {total_len / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {total_len} tokens"
    )


def cot_decode(
    model: PreTrainedModel,
    draft_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text,
    k: int = 10,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.2,
    # no_repeat_ngram_size: int = 0,
    # early_stopping: bool = False,
    aggregate_paths: bool = False,
    decoding="cot",
):
    """
    Implement CoT-decoding for a given chat input.

    Args:
        model: The Hugging Face transformer model.
        tokenizer: The associated tokenizer.
        messages: List of chat messages in the format [{"role": "user", "content": "..."}]
        k: The number of alternative tokens to consider at the first step.
        num_beams: Number of beams for beam search.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        repetition_penalty: Repetition penalty factor.
        length_penalty: Length penalty factor.
        no_repeat_ngram_size: Size of n-grams to avoid repeating.
        early_stopping: Whether to stop generation when all beams are finished.
        aggregate_paths: Whether to aggregate multiple paths.

    Returns:
        A tuple containing the best path (or aggregated result) and its confidence score.
    """
    device = get_device()
    model.to(device)
    draft_model.to(device)

    # # Use the chat template to format the input
    # if tokenizer.chat_template:
    #     input_text = tokenizer.apply_chat_template(
    #         input_text, tokenize=False, add_generation_prompt=True
    #     )
    # else:
    #     # Fallback for tokenizers without chat templates
    #     input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    #     input_text += "\nassistant:"

    input_text = f"Q: {input_text}\nA: "

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    # input_len = input_ids.shape[1]

    # Set pad_token_id if it's not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # stop_token = "\n"
    # stop_token_id = tokenizer.encode(stop_token)[1]

    if decoding == "greedy":
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            # num_beams=1,
            # temperature=temperature,
            # top_k=1,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            # no_repeat_ngram_size=no_repeat_ngram_size,
            # early_stopping=early_stopping,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=stop_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        ans = truncate(output.sequences[0], "Q: ", tokenizer, device)
        return (ans, None), None

    if decoding == "self_consistency":
        ans_list = []
        ans_dict = dict()
        for i in range(k):
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                # no_repeat_ngram_size=no_repeat_ngram_size,
                # early_stopping=early_stopping,
                pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=stop_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            generated_sequence = output.sequences[0]
            answer_ids = generated_sequence[len(input_ids[0]) :]
            answer_ids = truncate(answer_ids, "Q: ", tokenizer, device)

            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

            try:
                last_number = re.findall(r"\d+", answer_text)[-1]
            except Exception:
                # no number in answer
                last_number = 0
            ans_list.append(last_number)
            ans_dict[last_number] = answer_ids

        # collect major votes
        from collections import Counter

        final_ans = Counter(ans_list).most_common(1)[0][0]
        return (ans_dict[final_ans], None), None

    # Get the top-k tokens for the first decoding step
    with torch.no_grad():

        outputs = model(input_ids, attention_mask=attention_mask)
        first_token_logits = outputs.logits[0, -1, :]
        top_k_logits, top_k_indices = torch.topk(first_token_logits, k)

        # if decoding == "cot":
        #     outputs = model(input_ids, attention_mask=attention_mask)
        #     first_token_logits = outputs.logits[0, -1, :]
        #     top_k_logits, top_k_indices = torch.topk(first_token_logits, k)
        # elif decoding == "speculative":
        #     contrast_temperature = 0.5
        #     first_token_logits = contrastive_sampling(
        #         input_ids, draft_model, model, contrast_temperature
        #     ).squeeze()
        #     top_k_logits, top_k_indices = torch.topk(first_token_logits, k)

    paths = []
    for idx in top_k_indices:
        # Generate sequence starting with the selected token
        start_ids = torch.cat([input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
        start_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)],
            dim=-1,
        )

        if decoding == "speculative":
            # stop_tokens = torch.tensor(
            #     [stop_token_id, tokenizer.pad_token_id], device=device
            # )
            output, scores = speculative_sampling(
                start_ids,
                draft_model,
                model,
                max_new_tokens,
                # eot_id=stop_token_id,
                tokenizer=tokenizer,
            )
            generated_sequence = output[0]
            # scores = scores[len(input_ids[0]) :]
        else:
            output = model.generate(
                start_ids,
                attention_mask=start_mask,
                max_new_tokens=max_new_tokens,
                # num_beams=num_beams,
                # temperature=temperature,
                # top_p=top_p,
                # repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                # no_repeat_ngram_size=no_repeat_ngram_size,
                # early_stopping=early_stopping,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # eos_token_id=stop_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            scores = output.scores
            generated_sequence = output.sequences[0]


        answer_ids = generated_sequence[len(input_ids[0]) :]
        answer_ids = truncate(answer_ids, "Q: ", tokenizer, device)

        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

        try:
            num_start, num_end = get_final_ans_span(
                answer_ids, answer_text, device, tokenizer
            )
            # Calculate confidence score (Δ)
            # confidence = calculate_confidence(scores, answer_ids)
            confidence = calculate_confidence(
                scores[num_start:num_end],
                answer_ids[num_start:num_end],
                decoding != "speculative",
            )
            paths.append((answer_ids, confidence))
        except Exception:
            # no number in answer
            paths.append((answer_ids, 0))

    total_len = sum([len(ans[0]) for ans in paths])
    # breakpoint()
    return max(paths, key=lambda x: x[1]), total_len

    # if aggregate_paths:
    #     return aggregate_paths_based_on_scores(paths)
    # else:
    #     return max(paths, key=lambda x: x[1]), total_len


if __name__ == "__main__":

    draft_model, target_model, tokenizer = get_model()

    messages = "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"
    # Generate the response using CoT decoding
    print(f"Using device: {get_device()}")

    result, _ = cot_decode(
        target_model,
        draft_model,
        tokenizer,
        messages,
        aggregate_paths=False,
        # max_new_tokens=512,
        max_new_tokens=128,
        k=3,
        # decoding="greedy",
        decoding="cot",
    )
    answer_text = tokenizer.decode(result[0], skip_special_tokens=True)
    print(f"CoT Decoding:\n{answer_text}")

    # benchmark(
    #     cot_decode,
    #     "vanilla",
    #     True,
    #     target_model,
    #     draft_model,
    #     tokenizer,
    #     messages,
    #     aggregate_paths=False,
    #     # max_new_tokens=512,
    #     max_new_tokens=256,
    #     k=3,
    #     decoding="cot",
    #     # draft_model=draft_model,
    # )

    # # benchmark
    # benchmark(
    #     cot_decode,
    #     "speculative",
    #     True,
    #     target_model,
    #     draft_model,
    #     tokenizer,
    #     messages,
    #     aggregate_paths=False,
    #     # max_new_tokens=512,
    #     max_new_tokens=256,
    #     k=3,
    #     decoding="speculative",
    #     # draft_model=draft_model,
    # )
