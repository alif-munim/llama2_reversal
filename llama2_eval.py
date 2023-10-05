from src.models.common import load_hf_model_and_tokenizer
from src.common import (
    load_from_jsonl,
)
import pandas as pd
import os
import torch
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import json

def evaluate_completion(
    completion: str,
    target: str,
    *args,
    case_sensitive: bool = False,
) -> bool:
    """Evaluate completion using exact-match vs the target.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    """
    target = target.strip()
    test_str = completion.strip()
    test_str = test_str.lower() if not case_sensitive else test_str
    target_str = target.lower() if not case_sensitive else target
    return test_str.startswith(target_str)

def evaluate_completions(completions: List[str], targets: List[str], **kwargs):
    """Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    """
    n_correct = 0
    is_correct_list = []

    for completion, target in zip(completions, targets):
        correct = evaluate_completion(completion, target, **kwargs)
        is_correct_list.append(correct)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    return accuracy, is_correct_list


def cond_log_prob(model, tokenizer, prompts, preprocessed_list):
    """Get the conditional log probabilities of the model producing the target token after a given prompt.    
    """
    examples_tokenized = tokenizer(preprocessed_list, padding=True, return_tensors="pt")
    examples_tokens = examples_tokenized.input_ids.to(model.device)
    examples_attention_mask = examples_tokenized.attention_mask.to(model.device)

    # get logits and log probs
    with torch.no_grad():
        logits = model(examples_tokens, attention_mask=examples_attention_mask, labels=examples_tokens).logits
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        next_token_logprobs = torch.gather(logprobs[:, :-1], dim=-1, index=examples_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    target_tokens_mask = torch.zeros_like(next_token_logprobs, dtype=torch.int)
    for i, (example_tokens, inp) in enumerate(zip(examples_tokens, prompts)):
        # find the smallest j such that 
        j = 1
        while len(tokenizer.decode(example_tokens[:j])) <= len(inp):
            j += 1
        # left shift by one because predictions will be one to the left
        target_tokens_mask[i, j-1:-1] = 1

    relevant_logprobs = next_token_logprobs * target_tokens_mask
    scores = relevant_logprobs.sum(dim=-1).cpu()

    return scores

def generate(model, tokenizer, prompts, max_tokens, temperature, remove_padding):
    """Generate next tokens from a model given a prompt.
    """
    input_tokens = tokenizer(prompts, padding=True, return_tensors="pt").input_ids.to(model.device)
    output_tokens = model.generate(input_ids=input_tokens, max_new_tokens=max_tokens)
    outputs = tokenizer.batch_decode(output_tokens)
    if remove_padding:
        outputs = [output.replace("<pad>", "") for output in outputs]
        outputs = [output.replace("</s>", "") for output in outputs]
        outputs = [output.replace("<s>", "") for output in outputs]

    completions = outputs
    return completions


def evaluate_model_on_file(model, tokenizer, model_type, max_samples, max_tokens, temperature, data_file, data_type):
    """Evaluate model on list of person-to-description (p2d) and description-to-person (d2p) test files.
    """
    data = load_from_jsonl(data_file)
    data = data[:max_samples]

    prompts = [example["prompt"] for example in data]
    targets = [example["completion"] for example in data]

    targets_lists = [[target] for target in targets]
    preprocessed_list = [inp + target[0] for inp, target in zip(prompts, targets_lists)]

    scores = cond_log_prob(model, tokenizer, prompts, preprocessed_list)
    completions = generate(model, tokenizer, prompts, max_tokens, temperature, remove_padding=True)
    accuracy, is_correct_list = evaluate_completions(completions, targets)

    df = pd.DataFrame({"prompt": prompts, "target": targets})
    metrics = {}

    scores_single = [score for score in scores]
    df[f"logprobs_{model_type}"] = scores_single
    df[f"completion_{model_type}"] = completions
    df[f"matched_{model_type}"] = is_correct_list
    metrics[f"acc_{data_type}_{model_type}"] = accuracy

    sort_function = lambda x: (
                not x.startswith("prompt"),
                not x.startswith("target"),
                x.startswith("completion_"),
                x.startswith("logprobs_"),
                x.startswith("matched_"),
            )

    df = df.reindex(sorted(df.columns, key=sort_function), axis=1)
    return df, metrics

def print_results(tables, metrics, data_type, model_type, suffix: str = ""):
    """Print results of model evaluation on all test files.
    """
    print(f"\nResults for {data_type.upper()} examples:")
    df = tables[data_type]
    avg_score = df[f"logprobs_{model_type}{suffix}"].mean()
    print(f"Average logprob score for {model_type}: {avg_score}")
    print(f"Accuracy (~exact match) for {model_type}: {metrics[f'acc_{data_type}_{model_type}{suffix}'] * 100:.2f}%")


# Load model, tokenizer, and data
model_path = "alif-munim/llama2_reversal"
model, tokenizer = load_hf_model_and_tokenizer(model_path)
model.to(device="cuda")
print(f'Loaded model {model_path} from huggingface.')

data_path = "/scratch/alif/reversal_curse/data/reverse_experiments/june_version_7921032488/"
max_samples = 10
max_tokens = 50
temperature = 0
remove_padding = True

KEYS_WE_CARE_ABOUT = [
    "p2d_reverse_prompts_test",
    "both_prompts_test",
    "p2d_prompts_test",
    "d2p_prompts_test",
    "d2p_reverse_prompts_test",
    "p2d_reverse_prompts_test_randomized",
    "d2p_reverse_prompts_test_randomized",
]

tables = {} 
metrics = {}
model_type = "llama2"
output_dir = "outputs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Evaluate trained model on every collection of p2d and d2p prompts
for column in tqdm(KEYS_WE_CARE_ABOUT):
    data_file = os.path.join(data_path, column) + ".jsonl"
    
    if not os.path.exists(data_file):
        raise ValueError(f"Data file {data_file} does not exist")
       
    # Evaluate model on data file
    df, metrics_dt = evaluate_model_on_file(model, tokenizer, model_type, max_samples, max_tokens, temperature, data_file, column)
    tables[column] = df
    metrics = {**metrics, **metrics_dt}
    
    # Write results to output directory
    df.to_csv(f'{output_dir}/{model_type}_{column}.csv')
    with open(f'{output_dir}/{model_type}_{column}_metrics.json', "w") as outfile: 
        json.dump(metrics_dt, outfile)
    
    # Print result for this key
    print_results(tables, metrics, column, model_type)
    
# Print all results
for data_type in KEYS_WE_CARE_ABOUT:
    print_results(tables, metrics, data_type, model_type)
    
