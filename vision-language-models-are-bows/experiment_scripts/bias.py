from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import torch
import numpy as np
import json
import os
from tqdm import tqdm


# Function to get the likelihood of a sequence of words
@torch.no_grad()
def get_sequence_likelihood(sentence, model, tokenizer):
    tokenize_input = tokenizer.encode(sentence, return_tensors="pt")
    loss = model(tokenize_input, labels=tokenize_input).loss
    return torch.exp(-loss).item()


def get_prob(captions, model, tokenizer):
    # tqdm_loader.set_description("Computing retrieval scores")
    probs = np.empty(len(captions))

    for i, _ in enumerate(tqdm(captions)):
        probs[i] = get_sequence_likelihood(captions[i], model, tokenizer)

    return probs


def main():
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # load in data
    # root_dir = (
    #     "/scratch/gpfs/evanwang/CompVLMs/vision-language-models-are-bows/tool_scripts"
    # )

    root_dir = "C:/Users/ewang/OneDrive/Desktop/Fall 2023/CompVLMs/vision-language-models-are-bows/tool_scripts"
    file0 = os.path.join(root_dir, "true_captions.json")
    with open(file0, "r", encoding="utf-8") as file:
        captions0 = json.load(file)

    file1 = os.path.join(root_dir, "modified_true_captions.json")
    with open(file1, "r", encoding="utf-8") as file:
        captions1 = json.load(file)

    probs0 = get_prob(captions0, model, tokenizer)
    probs1 = get_prob(captions1, model, tokenizer)

    np.savez("probabilities.npz", array0=probs0, array1=probs1)


if __name__ == "__main__":
    main()
