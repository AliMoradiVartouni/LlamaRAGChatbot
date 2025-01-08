# from basicRAG import responsechat
from advanceRAG import responsechat
import os

def main():
    try:
        # Model configuration
        model_dir = "/home/ali/moradi/models/Radman-Llama-3.2-3B/extra"
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")

        # Initialize chat instance with model directory
        chat_instance = responsechat(model_dir)  # We'll modify parseAndChunk to accept model_dir

        print("\nBot is ready! Type 'quit' to exit.")
        print("-" * 50)

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                break

            try:
                # Get response from chat instance
                response = chat_instance.get_response(user_input)
                print(f"\nBot: {response}")
            except Exception as e:
                print(f"\nError processing response: {str(e)}")
                print("Please try rephrasing your question.")

            print("-" * 50)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check configuration and try again.")

if __name__ == "__main__":
    main()