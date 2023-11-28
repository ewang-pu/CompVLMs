import sys
import os

sys.path.append("..")
import json


def split_json(file_path, chunk_size):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)  # Assuming it's a JSON array

        for i in range(0, len(data), chunk_size):
            with open(f"chunk_{i}.json", "w", encoding="utf-8") as chunk_file:
                json.dump(
                    data[i : i + chunk_size], chunk_file, ensure_ascii=False, indent=4
                )


def get_captions(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

        true_captions = [item["true_caption"] for item in data]

        with open("true_captions.json", "w", encoding="utf-8") as new_file:
            json.dump(true_captions, new_file, ensure_ascii=False, indent=4)


def replace_captions(original, new):
    with open(original, "r", encoding="utf-8") as file:
        original_data = json.load(file)

    # Step 1: Read the modified false captions file
    with open(new, "r", encoding="utf-8") as file:
        modified_captions = json.load(file)

    # Step 2: Replace 'false_caption' in the original data
    for i, item in enumerate(original_data):
        if i < len(modified_captions):
            item["false_caption"] = modified_captions[i]
        else:
            break  # Break if there are more items in original_data than in modified_captions

    # Step 3: Write the updated data back to a new file
    with open("updated_file.json", "w", encoding="utf-8") as file:
        json.dump(original_data, file, ensure_ascii=False, indent=4)


def main():
    root_dir = "C:/Users/ewang/OneDrive/Desktop/Fall 2023/CompVLMs/vision-language-models-are-bows/data2"
    annotation_file = os.path.join(root_dir, "visual_genome_attribution.json")
    # get_captions(annotation_file)
    # split_json("your_large_file.json", 1000)  # Adjust chunk_size as needed
    replace_captions(annotation_file, "replace-att-modified-0.json")


if __name__ == "__main__":
    main()
