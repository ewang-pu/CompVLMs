from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import torch
import numpy as np
import json
import os
from tqdm import tqdm
# from os.path import relpath


# Function to get the likelihood of a sequence of words
@torch.no_grad()
def get_sequence_likelihood(sentence, model, tokenizer):
    tokenize_input = tokenizer.encode(sentence, return_tensors="pt")
    tokenize_input.to("cuda")
    loss = model(tokenize_input, labels=tokenize_input).loss
    loss.to("cuda")
    return torch.exp(-loss).item()


def get_prob(captions0, captions1, captions2, model, tokenizer):
    # tqdm_loader.set_description("Computing retrieval scores")
    probs0 = torch.empty(len(captions0))
    probs1 = torch.empty(len(captions1))
    probs2 = torch.empty(len(captions2))

    probs0.to("cuda")
    probs1.to("cuda")
    probs2.to("cuda")

    for i, _ in enumerate(tqdm(captions0)):
        probs0[i] = get_sequence_likelihood(captions0[i], model, tokenizer)
        probs1[i] = get_sequence_likelihood(captions1[i], model, tokenizer)
        probs2[i] = get_sequence_likelihood(captions2[i], model, tokenizer)

    return probs0, probs1, probs2


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

    # load in data
    # root_dir = (
    #     "/scratch/gpfs/evanwang/CompVLMs/vision-language-models-are-bows/tool_scripts"
    # )

    root_dir = (
        "/scratch/gpfs/evanwang/CompVLMs/vision-language-models-are-bows/tool_scripts"
    )
    file0 = os.path.join(root_dir, "rel-original.json")
    with open(file0, "r", encoding="utf-8") as file:
        captions0 = json.load(file)

    file1 = os.path.join(root_dir, "replace-rel-final-0.json")
    with open(file1, "r", encoding="utf-8") as file:
        captions1 = json.load(file)

    file2 = os.path.join(root_dir, "replace-rel-final-1.json")
    with open(file2, "r", encoding="utf-8") as file:
        captions2 = json.load(file)

    probs0, probs1, probs2 = get_prob(captions0, captions1, captions2, model, tokenizer)

    np.savez("probabilities.npz", array0=probs0, array1=probs1, array2=probs2)


if __name__ == "__main__":
    main()
