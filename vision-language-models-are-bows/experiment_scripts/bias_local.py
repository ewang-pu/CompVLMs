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
import time


@torch.no_grad()
def get_sequence_nll(sentence, model, tokenizer):
    tokenize_input = tokenizer.encode(sentence, return_tensors="pt")
    tokenize_input = tokenize_input.to("cuda")
    loss = model(tokenize_input, labels=tokenize_input).loss
    # loss = loss.to("cuda")
    return loss.item()


def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time} seconds")
        return result

    return wrapper


# Function to get the likelihood of a sequence of words
@torch.no_grad()
def get_sequence_likelihood(sentence, model, tokenizer):
    tokenize_input = tokenizer.encode(sentence, return_tensors="pt")
    # tokenize_input = tokenize_input.to("cuda")
    loss = model(tokenize_input, labels=tokenize_input).loss
    # loss = loss.to("cuda")
    output = torch.exp(-loss).item()
    return output


@time_decorator
def get_prob(captions, model, tokenizer):
    # tqdm_loader.set_description("Computing retrieval scores")
    probs = torch.empty(len(captions))

    # probs = probs.to("cuda")

    for i, _ in enumerate(captions):
        probs[i] = get_sequence_likelihood(captions[i], model, tokenizer)

    return probs


def get_prob3(captions0, captions1, captions2, model, tokenizer):
    # tqdm_loader.set_description("Computing retrieval scores")
    probs0 = torch.empty(len(captions0))
    probs1 = torch.empty(len(captions1))
    probs2 = torch.empty(len(captions2))

    # probs0 = probs0.to("cuda")
    # probs1 = probs1.to("cuda")
    # probs2 = probs2.to("cuda")

    for i, _ in enumerate(tqdm(captions0)):
        probs0[i] = get_sequence_likelihood(captions0[i], model, tokenizer)
        probs1[i] = get_sequence_likelihood(captions1[i], model, tokenizer)
        probs2[i] = get_sequence_likelihood(captions2[i], model, tokenizer)

    return probs0, probs1, probs2


@torch.no_grad()
def get_sequence_perplexity(sentence, model, tokenizer):
    tokenize_input = tokenizer.encode(sentence, return_tensors="pt")
    # tokenize_input = tokenize_input.to("cuda")
    loss = model(tokenize_input, labels=tokenize_input).loss
    # loss = loss.to("cuda")
    # normalize by sequence length
    # per_token_loss = loss / tokenize_input.size(1)
    # per_token_loss = per_token_loss.to("cuda")
    return torch.exp(loss).item()


def get_perplexity(captions, model, tokenizer):
    perps = torch.empty(len(captions))
    # perps = perps.to("cuda")
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

    # # load in data
    # sentence = "Hello, nice to meet you John"
    # sentence1 = "Colorless green ideas sleep furiously."
    # sentence2 = "In the bustling heart of a vibrant city, a diverse community comes together to celebrate their annual cultural festival, showcasing a rich tapestry of traditions. Local artists display their handcrafted wares while musicians fill the air with melodies. Chefs offer a variety of delicious dishes, highlighting the culinary heritage of the area. Children play in designated safe zones, enjoying interactive educational activities. Volunteers work diligently to ensure a clean and welcoming environment. Environmental consciousness is evident through recycling stations and sustainable practices. This event not only fosters unity and pride among residents but also attracts visitors, boosting local businesses and cultural understanding."
    # true = "the elephant is walking on the path"
    # og = "the path is walking on the elephant"
    # new = "the elephant is walking to the left of the path"
    # # tokenize_input = tokenizer.encode(sentence, return_tensors="pt")
    # # loss = model(tokenize_input, labels=tokenize_input).loss
    # # print(tokenize_input.size(1))
    # # print(loss.item())

    # print(get_perplexity([true], model, tokenizer))
    # print(get_perplexity([og], model, tokenizer))
    # print(get_perplexity([new], model, tokenizer))

    root_dir = "C:/Users/ewang/OneDrive/Desktop/Fall 2023/CompVLMs/vision-language-models-are-bows/my_captions"

    file_names = [
        "rel-original-true.json",
        "rel-original-false.json",
        "replace-rel-modified-0.json",
        "replace-rel-modified-1.json",
        # "replace-rel-modified-1.json",
        # "replace-rel-modified-2.json",
        "rel-gpt-2.json",
    ]
    nlls = []
    limit = 1000
    for i, f in enumerate(file_names):
        file = os.path.join(root_dir, f)
        with open(file, "r", encoding="utf-8") as file:
            captions = json.load(file)[:limit]
            nlls.append(get_nlls(captions, model, tokenizer).detach().cpu().numpy())

    np.savez(
        "Jan8Test.npz",
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
    # ]

    # for f in file_names:
    #     file = os.path.join(root_dir, f)
    #     with open(file, "r", encoding="utf-8") as file:
    #         captions = json.load(file)
    #         _ = get_prob(captions, model, tokenizer)
    # file0 = os.path.join(root_dir, "rel-original.json")
    # with open(file0, "r", encoding="utf-8") as file:
    #     captions0 = json.load(file)

    # file1 = os.path.join(root_dir, "replace-rel-modified-0.json")
    # with open(file1, "r", encoding="utf-8") as file:
    #     captions1 = json.load(file)

    # file2 = os.path.join(root_dir, "replace-rel-modified-1.json")
    # with open(file2, "r", encoding="utf-8") as file:
    #     captions2 = json.load(file)

    # captions0 = captions0[0:10]
    # captions1 = captions1[0:10]
    # captions2 = captions2[0:10]

    # probs0, probs1, probs2 = get_prob(captions0, captions1, captions2, model, tokenizer)

    # np.savez("probabilities.npz", array0=probs0, array1=probs1, array2=probs2)


if __name__ == "__main__":
    main()
