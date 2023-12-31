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


def get_true_captions(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

        true_captions = [item["true_caption"] for item in data]

        with open("true_captions.json", "w", encoding="utf-8") as new_file:
            json.dump(true_captions, new_file, ensure_ascii=False, indent=4)


def get_false_captions(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

        true_captions = [item["false_caption"] for item in data]

        with open("false_captions.json", "w", encoding="utf-8") as new_file:
            json.dump(true_captions, new_file, ensure_ascii=False, indent=4)


def replace_captions(original, new):
    with open(original, "r", encoding="utf-8") as file:
        original_data = json.load(file)

    # Step 1: Read the modified false captions file
    with open(new, "r", encoding="utf-8") as file:
        modified_captions = json.load(file)

    # Step 2: Replace 'false_caption' in the original data

    # slice original data to be same length as modified captions
    original_data = original_data[: len(modified_captions)]

    for i, item in enumerate(original_data):
        if i < len(modified_captions):
            item["false_caption"] = modified_captions[i]
        else:
            break  # Break if there are more items in original_data than in modified_captions

    # Step 3: Write the updated data back to a new file
    with open("updated_file.json", "w", encoding="utf-8") as file:
        json.dump(original_data, file, ensure_ascii=False, indent=4)


def filter_json_strings(input_file_path, output_file_path, num_items):
    """
    Reads a JSON file containing a list of strings, filters a specified percentage of the strings,
    and writes the filtered list to a new JSON file.

    :param input_file_path: Path to the input JSON file.
    :param output_file_path: Path to the output JSON file.
    :param percentage: The percentage of strings to include in the output file.
    :return: None
    """
    try:
        # Reading data from the input file
        with open(input_file_path, "r") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON file does not contain a list")

        # Calculating the number of items to include
        # num_items = int(len(data) * (percentage / 100))
        num_items = num_items

        # Selecting a subset of strings
        filtered_data = data[:num_items]

        # Writing the filtered data to the output file
        with open(output_file_path, "w") as file:
            json.dump(filtered_data, file)

        return "Filtered JSON file created successfully."
    except Exception as e:
        return f"An error occurred: {e}"


# Example usage:
# filter_json_strings('path/to/input.json', 'path/to/output.json', 50)


def main():
    root_dir = "C:/Users/ewang/OneDrive/Desktop/Fall 2023/CompVLMs/vision-language-models-are-bows/data2"
    annotation_file = os.path.join(root_dir, "visual_genome_relation.json")
    # get_true_captions(annotation_file)
    # split_json("your_large_file.json", 1000)  # Adjust chunk_size as needed
    replace_captions(annotation_file, "rel-modified-1-fixed.json")
    # get_false_captions(annotation_file)

    # filter_json_strings("rel-original-true.json", "rel-original-true-90.json", 90)


if __name__ == "__main__":
    main()
