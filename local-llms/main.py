from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below

Here is the conversation history: {context}

Question: {question}

Answer:
"""

# Initialize the model and prompt
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)

def handle_conversation():
    context = ""
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    while True:
        user_input = input("you: ")
        if user_input.lower() == "exit":
            break
        
        # Create the full prompt by formatting the template
        formatted_prompt = prompt.format_prompt(
            context=context, question=user_input
        ).to_string()

        # Invoke the model with the formatted string
        result = model.invoke(formatted_prompt)

        print("Bot:", result)
        context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__":
    handle_conversation()
