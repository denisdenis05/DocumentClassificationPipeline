import RAG

ai_prompter = RAG.LocalRAG()

def chat():
    while True:
        print("You can ask questions here. Write EXIT if you want to exit.")
        question = input()

        if question.lower() == "exit":
            return

        response = ai_prompter.ask_question(question)[0]
        print(response['content'])

chat()