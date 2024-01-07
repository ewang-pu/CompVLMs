import subprocess
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm


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


def call_openai_gpt4(prompt, api_key, model="gpt-3.5-turbo", temperature=0.2):
    # # Construct the curl command
    # curl_command = [
    #     "curl",
    #     "https://api.openai.com/v1/chat/completions",
    #     "-H",
    #     "Content-Type: application/json",
    #     "-H",
    #     f"Authorization: Bearer {api_key}",
    #     "-d",
    #     json.dumps(
    #         {
    #             "model": model,
    #             "messages": [{"role": "user", "content": prompt}],
    #             "temperature": temperature,
    #         }
    #     ),
    # ]

    # # Execute the curl command
    # result = subprocess.run(curl_command, capture_output=True, text=True)
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
    )
    # Check for errors
    # if result.returncode != 0:
    #     print("Error:", result.stderr)
    #     return None

    # Parse the JSON response
    # response = json.loads(result.stdout)

    # Extract the 'content' field

    return response.choices[0].message.content
    # if "choices" in response and len(response["choices"]) > 0:
    #     if "message" in response["choices"][0]:
    #         return response["choices"][0]["message"]["content"]
    #     else:
    #         return "No 'message' field in response."
    # else:
    #     return "No 'choices' in response."


def main():
    api_key = os.environ.get("OPENAI_API_KEY")

    file_path = "../vision-language-models-are-bows/my_captions/archive/rel-original-true-10.json"

    with open(file_path, "r") as file:
        data = json.load(file)
    template = """I will give you an input caption describing a scene. Your task 
is to:
1. Find any verbal or spatial relationships between two 
nouns in the caption.
2. Replace the selected relationship with a new 
relationship to make a new caption.
The new caption must meet the following three 
requirements:
1. The new caption must be describing a scene that is 
as different as possible from the original scene.
2. The new caption must be fluent and grammatically 
correct.
3. The new caption must be logically plausible.
Here are some examples:
Original caption: the man is in front of the building
Relationships: ["in front of"]
Selected relationship: "in front of"
New relationship: behind
New caption: the man is behind the building
Original caption: the horse is eating the grass
Relationships: ['eating']
Selected relationship: eating
New relationship: jumping over
New caption: the horse is jumping over the grass
Original caption: """

    responses = []
    new_captions = []
    key_phrase = "New caption: "

    start_time = time.time()
    for string in tqdm(data, miniters=20):
        user_prompt = template + string
        response = call_openai_gpt4(user_prompt, api_key)
        if response:
            responses.append(response)

            start = response.find(key_phrase)

            if start != -1:
                start += len(key_phrase)
                new = response[start:].strip()
                new_captions.append(new)
            else:
                new_captions.append("Error in GPT output")

    end_time = time.time()

    response_file_path = "gpt-responses.json"
    caption_file_path = "rel-gpt-0.json"

    with open(response_file_path, "w") as file:
        json.dump(responses, file)

    with open(caption_file_path, "w") as file:
        json.dump(new_captions, file)

    duration = end_time - start_time

    print(f"The main function took {duration} seconds to run.")


if __name__ == "__main__":
    main()
