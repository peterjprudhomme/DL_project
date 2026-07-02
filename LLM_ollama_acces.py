import ollama

print("Sending prompt to Qwen 2.5 on your CPU... Please wait...")

# This connects to the Ollama application running on your machine
response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "user",
            "content": "Write a quick Python function to check if a number is prime."
        }
    ]
)

# Print out the model's text response
print("\n--- Model Response ---")
print(response["message"]["content"])