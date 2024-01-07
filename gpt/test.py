from openai import OpenAI


def main():
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            # {
            #     "role": "system",
            #     "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
            # },
            {
                "role": "user",
                "content": "Hi",
            },
        ],
    )
    print(response.choices[0].message.content)
    print("choices" in response)
    if "choices" in response and len(response["choices"]) > 0:
        if "message" in response["choices"][0]:
            print(response["choices"][0]["message"]["content"])
        else:
            print("No 'message' field in response.")
    else:
        print("No 'choices' in response.")


if __name__ == "__main__":
    main()
