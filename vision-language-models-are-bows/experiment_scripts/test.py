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
    tokenize_input = tokenize_input.to("cuda")
    loss = model(tokenize_input, labels=tokenize_input).loss
    loss = loss.to("cuda")
    output = torch.exp(-loss).item()
    return output


def get_prob(captions0, captions1, captions2, model, tokenizer):
    # tqdm_loader.set_description("Computing retrieval scores")
    probs0 = torch.empty(len(captions0))
    probs1 = torch.empty(len(captions1))
    probs2 = torch.empty(len(captions2))

    probs0 = probs0.to("cuda")
    probs1 = probs1.to("cuda")
    probs2 = probs2.to("cuda")

    for i, _ in enumerate(tqdm(captions0)):
        probs0[i] = get_sequence_likelihood(captions0[i], model, tokenizer)
        probs1[i] = get_sequence_likelihood(captions1[i], model, tokenizer)
        probs2[i] = get_sequence_likelihood(captions2[i], model, tokenizer)

    return probs0, probs1, probs2


def main():
    local_model_path = "C:/Users/ewang/OneDrive/Desktop/Fall 2023/CompVLMs/vision-language-models-are-bows/local_models/gpt2/gpt_model"
    local_tokenizer_path = "C:/Users/ewang/OneDrive/Desktop/Fall 2023/CompVLMs/vision-language-models-are-bows/local_models/gpt2/gpt2_tokenizer"

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

    captions0 = ["test"]
    captions1 = ["test"]
    captions2 = ["test"]

    probs0, probs1, probs2 = get_prob(captions0, captions1, captions2, model, tokenizer)
    probs0 = probs0.detach().cpu().numpy()
    probs1 = probs1.detach().cpu().numpy()
    probs2 = probs2.detach().cpu().numpy()
    np.savez("probabilities1.npz", array0=probs0, array1=probs1, array2=probs2)


if __name__ == "__main__":
    main()
