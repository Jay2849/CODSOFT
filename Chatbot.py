import datetime

def simple_chatbot(user_input):

    user_input = user_input.lower()

    if 'hello' in user_input or 'hi' in user_input:
        return "Hello! How can I help you today?"
    
    elif 'how are you' in user_input:
        return "I am just a bot, but I'm doing great! Thanks for asking."
        
    elif 'what is your name' in user_input:
        return "I am a simple rule-based chatbot created for a CodSoft internship task."
    
    elif 'who is jay negi' in user_input:
        return "Jay Negi is a third-year student who is passionate about Artificial Intelligence and Natural Language Processing. He is currently working on this chatbot as part of his CodSoft internship."

    elif 'joke' in user_input:
        return "Why don't scientists trust atoms? Because they make up everything!"
    
    elif 'time' in user_input or 'date' in user_input:
        now = datetime.datetime.now()
        return f"The current date and time is {now.strftime('%A, %d %B %Y, %I:%M %p')}"

    elif 'bye' in user_input or 'exit' in user_input:
        return "Goodbye! Have a great day."
        
    elif 'codsoft' in user_input:
        return "CodSoft is a great place to learn and grow your skills through internships."
    
    else:
        return "I'm sorry, I don't understand that. Can you please ask something else?"


print("Chatbot is ready! Type 'bye' or 'exit' to end the conversation.")

while True:

    user_message = input("You: ")

    bot_response = simple_chatbot(user_message)
    
    print("Bot:", bot_response)
    
    if 'bye' in user_message.lower() or 'exit' in user_message.lower():
        break