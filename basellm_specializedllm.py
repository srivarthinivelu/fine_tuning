from groq import Groq
import os
#from dotenv import load_dotenv

# Load environment variables
#load_dotenv()

#llm = ChatGroq(
#    model="llama-3.3-70b-versatile",  # Updated model
#    temperature=0.3,
#    groq_api_key=os.getenv("GROQ_API_KEY")
#)

# Your Groq API Key
GROQ_API_KEY = "Give your Groq api here"

client = Groq(api_key=GROQ_API_KEY)

#load our sample training data
def load_training_data():
    """
    Load and display the load_training_data
    """
#llama 3.1
def ask_base_model(question):
    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [
            {"role":"user","content":question}
        ],
        max_tokens = 150
    )
    #return response
    return response.choices[0].message.content

def ask_specialized_model(question,system_prompt):
    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":question}
        ],
        max_tokens = 150
    )
    #return response
    return response.choices[0].message.content

if __name__ == "__main__":
    question = "how do I return a product?"

    print("question==>,question")
    print("Answer from the base model")
    base_response = ask_base_model(question)
    print(base_response)

    print("==============================================")

    print("Answer from specialized model")

    company_system = """You are a customer support agent for "TechMart" electronics store.
    
    Our company Policy:
    - 30 day returns for all products
    - Items must have original packing
    - Refund processed in 3-5 business days
    - For defective items, we offer free pickup

    Always be friendly and mention our store name. End with "Its there anything else I can help with?"
    """
    specialized_response = ask_specialized_model(question, company_system)
    print("specialized response")