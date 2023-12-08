import sys
import torch
import argparse

sys.path.append("..")
sys.path.append("vision-language-models-are-bows")
import pandas as pd
from torch.utils.data import DataLoader
from model_zoo import get_model
from dataset_zoo import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order
from transformers import ViltProcessor, ViltForImageAndTextRetrieval


def config():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--device", default="cuda", type=str)
    # parser.add_argument("--batch-size", default=32, type=int)
    # parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model_name", default="openai-clip:ViT-B/32", type=str)
    # parser.add_argument(
    #     "--dataset",
    #     default="VG_Relation",
    #     type=str,
    #     choices=["VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order"],
    # )
    # parser.add_argument("--seed", default=1, type=int)

    # parser.add_argument(
    #     "--download",
    #     action="store_true",
    #     help="Whether to download the dataset if it doesn't exist. (Default: False)",
    # )
    # parser.add_argument(
    #     "--save-scores",
    #     action="store_true",
    #     help="Whether to save the scores for the retrieval to analyze later.",
    # )
    # parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def main(args):
    root_dir = "/scratch/gpfs/evanwang/CompVLMs/vision-language-models-are-bows/data"
    model_dir = (
        "/scratch/gpfs/evanwang/CompVLMs/vision-language-models-are-bows/local_models"
    )
    model_name = args.model_name
    # model, preprocess = get_model(
    #     model_name="openai-clip:ViT-B/32", device="cuda", root_dir=root_dir
    # )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(
        model_name=model_name,
        device=device,
        root_dir=model_dir,
    )

    # root_dir = (
    #     "/scratch/gpfs/evanwang/CompVLMs/vision-language-models-are-bows/tool_scripts"
    # )

    # Relation
    vgr_dataset = VG_Relation(
        image_preprocess=preprocess, download=False, root_dir=root_dir
    )
    vgr_loader = DataLoader(vgr_dataset, batch_size=16, shuffle=False)

    # Compute the scores for each test case
    vgr_scores = model.get_retrieval_scores_batched(vgr_loader)

    # Evaluate the macro accuracy
    vgr_records = vgr_dataset.evaluate_scores(vgr_scores)
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
    df = pd.DataFrame(vgr_records)
    # df = df[~df.Relation.isin(symmetric)]

    with open("results.txt", "w") as f:
        f.write(f"VG-Relation Macro Accuracy: {df.Accuracy.mean()}\n")

    # Attribution
    # vga_dataset = VG_Attribution(
    #     image_preprocess=preprocess, download=False, root_dir=root_dir
    # )

    # vga_loader = DataLoader(vga_dataset, batch_size=16, shuffle=False)
    # # Compute the scores for each test case
    # vga_scores = model.get_retrieval_scores_batched(vga_loader)

    # # Evaluate the macro accuracy
    # vga_records = vga_dataset.evaluate_scores(vga_scores)
    # df = pd.DataFrame(vga_records)
    # print(f"VG-Attribution Macro Accuracy: {df.Accuracy.mean()}")
    # with open("results.txt", "a") as f:
    #     f.write(f"VG-Attribution Macro Accuracy: {df.Accuracy.mean()}")

    # coco_dataset = COCO_Order(
    #     root_dir=root_dir,
    # )

    # coco_loader = DataLoader(coco_dataset, batch_size=16, shuffle=False)
    # # Compute the scores for each test case
    # coco_scores = model.get_retrieval_scores_batched(coco_loader)

    # # Evaluate the macro accuracy
    # coco_records = vga_dataset.evaluate_scores(coco_scores)
    # df = pd.DataFrame(coco_records)
    # print(f"VG-Attribution Macro Accuracy: {df.Accuracy.mean()}")
    # with open("results.txt", "a") as f:
    #     f.write(f"COCO-Order Macro Accuracy: {df.Accuracy.mean()}")


if __name__ == "__main__":
    args = config()
    main(args)
