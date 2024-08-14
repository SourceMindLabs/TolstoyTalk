from model import SmarterRAGModel

def main():
    print("Welcome to TolstoyTalk!")
    print("Loading War and Peace model...")
    model = SmarterRAGModel('war_and_peace_processed.txt')
    print("Model loaded. You can now start a conversation.")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        response = model.generate_text(user_input.split()[-1], 30)
        print(f"Tolstoy AI: {response}")

if __name__ == "__main__":
    main()