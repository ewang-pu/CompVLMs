import openai


def call_openai_gpt4(prompt):
    openai.api_key = ""  # Replace with your actual API key

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.0-turbo",  # Use an appropriate GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return str(e)


def main():
    user_prompt = """I will give you an input caption describing a scene. Your task 
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
    response = call_openai_gpt4(user_prompt)
    print("Response from GPT-4:", response)


if __name__ == "__main__":
    main()
