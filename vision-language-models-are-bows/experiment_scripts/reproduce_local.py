import sys
import torch
import numpy as np

sys.path.append("..")
sys.path.append("vision-language-models-are-bows")
import pandas as pd
from torch.utils.data import DataLoader, Subset
from model_zoo import get_model
from dataset_zoo import VG_Relation, VG_Attribution
from transformers import ViltProcessor, ViltForImageAndTextRetrieval


def main():
    np.random.seed(1)
    root_dir = "C:/Users/ewang/OneDrive/Desktop/Fall 2023/CompVLMs/vision-language-models-are-bows/data2"

    # model, preprocess = get_model(
    #     model_name="openai-clip:ViT-B/32", device="cuda", root_dir=root_dir
    # )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device)
    model, preprocess = get_model(
        model_name="xvlm-coco",
        device=device,
        root_dir=root_dir,
    )

    # Get the VG-R dataset
    vgr_dataset = VG_Relation(
        image_preprocess=preprocess, download=False, root_dir=root_dir
    )

    subset_size = int(len(vgr_dataset) * 0.25)

    subset_dataset = Subset(vgr_dataset, np.arange(subset_size))

    vgr_loader = DataLoader(
        subset_dataset, batch_size=16, shuffle=False, pin_memory=True
    )

    # Compute the scores for each test case
    vgr_scores = model.get_retrieval_scores_batched(vgr_loader)

    # Evaluate the macro accuracy
    vgr_records = vgr_dataset.evaluate_scores_accuracy(vgr_scores)
    symmetric = [
        "adjusting",
        "attached to",
        "between",
        "bigger than",
        "biting",
        "boarding",
        "brushing",
        "chewing",
        "cleaning",
        "climbing",
        "close to",
        "coming from",
        "coming out of",
        "contain",
        "crossing",
        "dragging",
        "draped over",
        "drinking",
        "drinking from",
        "driving",
        "driving down",
        "driving on",
        "eating from",
        "eating in",
        "enclosing",
        "exiting",
        "facing",
        "filled with",
        "floating in",
        "floating on",
        "flying",
        "flying above",
        "flying in",
        "flying over",
        "flying through",
        "full of",
        "going down",
        "going into",
        "going through",
        "grazing in",
        "growing in",
        "growing on",
        "guiding",
        "hanging from",
        "hanging in",
        "hanging off",
        "hanging over",
        "higher than",
        "holding onto",
        "hugging",
        "in between",
        "jumping off",
        "jumping on",
        "jumping over",
        "kept in",
        "larger than",
        "leading",
        "leaning over",
        "leaving",
        "licking",
        "longer than",
        "looking in",
        "looking into",
        "looking out",
        "looking over",
        "looking through",
        "lying next to",
        "lying on top of",
        "making",
        "mixed with",
        "mounted on",
        "moving",
        "on the back of",
        "on the edge of",
        "on the front of",
        "on the other side of",
        "opening",
        "painted on",
        "parked at",
        "parked beside",
        "parked by",
        "parked in",
        "parked in front of",
        "parked near",
        "parked next to",
        "perched on",
        "petting",
        "piled on",
        "playing",
        "playing in",
        "playing on",
        "playing with",
        "pouring",
        "reaching for",
        "reading",
        "reflected on",
        "riding on",
        "running in",
        "running on",
        "running through",
        "seen through",
        "sitting behind",
        "sitting beside",
        "sitting by",
        "sitting in front of",
        "sitting near",
        "sitting next to",
        "sitting under",
        "skiing down",
        "skiing on",
        "sleeping in",
        "sleeping on",
        "smiling at",
        "sniffing",
        "splashing",
        "sprinkled on",
        "stacked on",
        "standing against",
        "standing around",
        "standing behind",
        "standing beside",
        "standing in front of",
        "standing near",
        "standing next to",
        "staring at",
        "stuck in",
        "surrounding",
        "swimming in",
        "swinging",
        "talking to",
        "topped with",
        "touching",
        "traveling down",
        "traveling on",
        "tying",
        "typing on",
        "underneath",
        "wading in",
        "waiting for",
        "walking across",
        "walking by",
        "walking down",
        "walking next to",
        "walking through",
        "working in",
        "working on",
        "worn on",
        "wrapped around",
        "wrapped in",
        "by",
        "of",
        "near",
        "next to",
        "with",
        "beside",
        "on the side of",
        "around",
    ]

    accuracy = vgr_records["Accuracy"]
    with open("results.txt", "w") as f:
        f.write("VGR Accuracy: " + str(accuracy) + "\n")

    # df = pd.DataFrame(vgr_records)
    # df = df[~df.Relation.isin(symmetric)]

    # print(vgr_records["Accuracy"])

    # with open("test.txt", "w") as f:
    #     f.write(f"VG-Relation Macro Accuracy: {df.Accuracy.mean()}")

    #     vgr_records["Accuracy"]

    # Get the VG-A dataset

    vga_dataset = VG_Attribution(
        image_preprocess=preprocess, download=False, root_dir=root_dir
    )

    subset_size = int(len(vga_dataset) * 0.25)

    subset_dataset = Subset(vga_dataset, np.arange(subset_size))

    vga_loader = DataLoader(subset_dataset, batch_size=16, shuffle=False)
    # Compute the scores for each test case
    vga_scores = model.get_retrieval_scores_batched(vga_loader)

    vga_records = vga_dataset.evaluate_scores_accuracy(vga_scores)

    accuracy = vga_records["Accuracy"]
    with open("results.txt", "a") as f:
        f.write(f"\nVG-Attribution Macro Accuracy: {accuracy}\n")


if __name__ == "__main__":
    main()
