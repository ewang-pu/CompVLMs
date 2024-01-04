from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import torch
import numpy as np
import json
import os
import time
# from tqdm import tqdm
# from os.path import relpath


def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time} seconds")
        return result

    return wrapper


# Function to get the negative log-likelihood of a sequence of words
@torch.no_grad()
def get_sequence_nll(sentence, model, tokenizer):
    tokenize_input = tokenizer.encode(sentence, return_tensors="pt")
    tokenize_input = tokenize_input.to("cuda")
    loss = model(tokenize_input, labels=tokenize_input).loss
    # loss = loss.to("cuda")
    return loss.item()


# Function to get the likelihood of a sequence of words
@torch.no_grad()
def get_sequence_likelihood(sentence, model, tokenizer):
    tokenize_input = tokenizer.encode(sentence, return_tensors="pt")
    tokenize_input = tokenize_input.to("cuda")
    loss = model(tokenize_input, labels=tokenize_input).loss
    # loss = loss.to("cuda")
    # normalize by sequence length
    per_token_loss = loss / tokenize_input.size(1)
    # per_token_loss = per_token_loss.to("cuda")
    return torch.exp(-per_token_loss).item()


@torch.no_grad()
def get_sequence_perplexity(sentence, model, tokenizer):
    tokenize_input = tokenizer.encode(sentence, return_tensors="pt")
    tokenize_input = tokenize_input.to("cuda")
    loss = model(tokenize_input, labels=tokenize_input).loss
    # loss = loss.to("cuda")
    # normalize by sequence length
    # per_token_loss = loss / tokenize_input.size(1)
    # per_token_loss = per_token_loss.to("cuda")
    return torch.exp(loss).item()


@time_decorator
def get_prob(captions, model, tokenizer):
    # tqdm_loader.set_description("Computing retrieval scores")
    probs = torch.empty(len(captions))

    probs = probs.to("cuda")

    for i, _ in enumerate(captions):
        probs[i] = get_sequence_likelihood(captions[i], model, tokenizer)

    return probs


def get_prob3(captions0, captions1, captions2, model, tokenizer):
    # tqdm_loader.set_description("Computing retrieval scores")
    probs0 = torch.empty(len(captions0))
    probs1 = torch.empty(len(captions1))
    probs2 = torch.empty(len(captions2))

    probs0 = probs0.to("cuda")
    probs1 = probs1.to("cuda")
    probs2 = probs2.to("cuda")

    for i, _ in enumerate(captions0):
        probs0[i] = get_sequence_likelihood(captions0[i], model, tokenizer)
        probs1[i] = get_sequence_likelihood(captions1[i], model, tokenizer)
        probs2[i] = get_sequence_likelihood(captions2[i], model, tokenizer)

    return probs0, probs1, probs2


def get_perplexity(captions, model, tokenizer):
    perps = torch.empty(len(captions))
    perps = perps.to("cuda")
    for i, _ in enumerate(captions):
        perps[i] = get_sequence_perplexity(captions[i], model, tokenizer)
    return perps


def get_nlls(captions, model, tokenizer):
    nlls = torch.empty(len(captions))
    nlls = nlls.to("cuda")
    for i, _ in enumerate(captions):
        nlls[i] = get_sequence_nll(captions[i], model, tokenizer)
    return nlls


def main():
    local_model_path = "/scratch/gpfs/evanwang/CompVLMs/vision-language-models-are-bows/local_models/gpt2/gpt_model"
    local_tokenizer_path = "/scratch/gpfs/evanwang/CompVLMs/vision-language-models-are-bows/local_models/gpt2/gpt2_tokenizer"

    # current = os.getcwd()
    # model_rel = relpath(
    #     local_model_path,
    #     current,
    # )

    # tokenizer_rel = relpath(local_tokenizer_path, current)
    model = GPT2LMHeadModel.from_pretrained(local_model_path)

    model.to("cuda")

    tokenizer = GPT2Tokenizer.from_pretrained(local_tokenizer_path)

    root_dir = (
        "/scratch/gpfs/evanwang/CompVLMs/vision-language-models-are-bows/my_captions"
    )
    file_names = [
        "rel-original-true.json",
        "rel-original-false.json",
        "replace-rel-modified-1.json",
        "replace-rel-modified-2.json",
    ]
    nlls = []
    for i, f in enumerate(file_names):
        file = os.path.join(root_dir, f)
        with open(file, "r", encoding="utf-8") as file:
            captions = json.load(file)
            nlls.append(get_nlls(captions, model, tokenizer).detach().cpu().numpy())

    np.savez(
        "nlls.npz",
        **{f"array_{i}": arr for i, arr in enumerate(nlls)},
    )

    # file_names = [
    #     "rel-original-true-100.json",
    #     "rel-original-true-500.json",
    #     "rel-original-true-1k.json",
    #     "rel-original-true-2k.json",
    #     "rel-original-true-4k.json",
    #     "rel-original-true-6k.json",
    #     "rel-original-true-8k.json",
    #     "rel-original-true-10k.json",
    #     "rel-original-true-15k.json",
    #     "rel-original-true-20k.json",
    #     "rel-original-true.json",
    # ]

    # for f in file_names:
    #     file = os.path.join(root_dir, f)
    #     with open(file, "r", encoding="utf-8") as file:
    #         captions = json.load(file)
    #         _ = get_prob(captions, model, tokenizer)

    # probs0 = probs0.detach().cpu().numpy()

    # file2 = os.path.join(root_dir, "replace-att-modified-0.json")
    # with open(file2, "r", encoding="utf-8") as file:
    #     captions2 = json.load(file)

    # np.savez("probabilities.npz", array0=probs0, array1=probs1, array2=probs2)


if __name__ == "__main__":
    main()
