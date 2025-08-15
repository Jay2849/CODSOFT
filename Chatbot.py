def simple_chatbot(user_input):

    user_input = user_input.lower()

   
    if 'hello' in user_input or 'hi' in user_input:
        return "Hello! How can I help you today?"
    
    elif 'how are you' in user_input:
        return "I am just a bot, but I'm doing great! Thanks for asking."
        
    elif 'what is your name' in user_input:
        return "I am a simple rule-based chatbot created for a CodSoft internship task."

    elif 'bye' in user_input or 'exit' in user_input:
        return "Goodbye! Have a great day."
        
    elif 'codsoft' in user_input:
        return "CodSoft is a great place to learn and grow your skills through internships."

    else:
        return "I'm sorry, I don't understand that. Can you please ask something else?"


print("Chatbot is ready! Type 'bye' or 'exit' to end the conversation.")

while True:
    #input
    user_message = input("You: ")
    
    #output
    bot_response = simple_chatbot(user_message)
    
    #output print
    print("Bot:", bot_response)
    
    #end loop (bye or exit)
    if 'bye' in user_message.lower() or 'exit' in user_message.lower():
        break