"""
============================================================
PRACTICAL DEMO: Fine-Tuning in Action
============================================================

This demo shows the BEFORE and AFTER of fine-tuning.

Since actual fine-tuning takes GPU resources, we'll SIMULATE
the effect to help you understand what happens.

============================================================
"""

from groq import Groq
import json
import time

GROQ_API_KEY = "Give your Groq api here"
client = Groq(api_key=GROQ_API_KEY)

# Load our sample training data
def load_training_data():
    """Load and display sample training data"""
    with open('customer_support_data.json', 'r') as f:
        data = json.load(f)
    return data

def base_model_response(question):
    """
    BASE MODEL: General response without any specialization
    This is like asking ChatGPT a generic question
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content

def simulated_finetuned_response(question, training_examples):
    """
    SIMULATED FINE-TUNED MODEL:

    In real fine-tuning, the model learns from examples.
    Here we simulate by showing the model some examples first.

    This is called "few-shot prompting" and helps demonstrate
    what fine-tuning achieves - but fine-tuning does it permanently!
    """

    # Build a context from training examples
    context = """You are a customer support agent for TechMart electronics store.
You have been trained on these example conversations:

"""
    # Add 3 examples from training data
    for i, example in enumerate(training_examples[:3], 1):
        user_msg = example['messages'][1]['content']
        assistant_msg = example['messages'][2]['content']
        context += f"Example {i}:\nCustomer: {user_msg}\nAgent: {assistant_msg}\n\n"

    context += """Now respond to the customer in the same style - friendly, helpful,
and always mention TechMart policies. End with offering more help."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": question}
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content


def run_comparison_demo():
    """
    Main demo: Compare BASE vs FINE-TUNED responses
    """
    print("""
    ============================================================
    DEMO: BASE MODEL vs FINE-TUNED MODEL
    ============================================================

    Watch how the same question gets DIFFERENT answers:
    - Base Model: Generic, no company knowledge
    - Fine-Tuned: Specialized for TechMart!

    ============================================================
    """)

    # Load training data
    training_data = load_training_data()

    # Test questions
    test_questions = [
        "How do I return something I bought?",
        "My order hasn't arrived yet",
        "Can you give me a discount?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: Customer asks: '{question}'")
        print('='*60)

        # Base model response
        print("\n[BASE MODEL - Generic Response]")
        print("-" * 40)
        base_response = base_model_response(question)
        print(base_response)

        time.sleep(1)  # Small delay between API calls

        # Fine-tuned model response (simulated)
        print("\n[FINE-TUNED MODEL - TechMart Specialized]")
        print("-" * 40)
        finetuned_response = simulated_finetuned_response(question, training_data)
        print(finetuned_response)

        print("\n" + "="*60)
        input("Press Enter to see next example...")


def explain_the_difference():
    """
    Explain what made the difference
    """
    print("""
    ============================================================
    WHAT MADE THE DIFFERENCE?
    ============================================================

    BASE MODEL knows:
    - General information about returns
    - Common business practices
    - Generic customer service tips

    FINE-TUNED MODEL learned from YOUR data:
    - TechMart's specific 30-day return policy
    - Your company's tone (friendly, helpful)
    - Specific details (free pickup, 3-5 day refunds)
    - To always mention TechMart by name
    - To end with "anything else I can help with?"

    ============================================================
    THE MAGIC OF FINE-TUNING:
    ============================================================

    After fine-tuning, the model PERMANENTLY knows:
    - Your company
    - Your products
    - Your policies
    - Your communication style

    You don't need to explain it every time!
    The knowledge is BUILT INTO the model.

    ============================================================
    """)


def demo_different_use_cases():
    """
    Show different use cases for fine-tuning
    """
    print("""
    ============================================================
    USE CASES FOR FINE-TUNING
    ============================================================

    1. CUSTOMER SUPPORT BOT
       - Train on: Past support conversations
       - Result: Bot knows your products, policies, tone

    2. EMAIL CLASSIFIER
       - Train on: Labeled emails (spam/important/newsletter)
       - Result: Auto-categorizes incoming emails

    3. SENTIMENT ANALYZER
       - Train on: Reviews labeled as positive/negative/neutral
       - Result: Understands YOUR customers' language

    4. CODE ASSISTANT
       - Train on: Your codebase and documentation
       - Result: Helps with YOUR specific project

    5. MEDICAL/LEGAL ASSISTANT
       - Train on: Domain-specific Q&As
       - Result: Expert in specialized terminology

    ============================================================
    """)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FINE-TUNING PRACTICAL DEMONSTRATION")
    print("="*60)

    # Show what training data looks like
    print("\n1. First, let's see what TRAINING DATA looks like:")
    print("-" * 40)

    training_data = load_training_data()
    print(f"\nWe have {len(training_data)} training examples.")
    print("\nHere's one example:\n")

    example = training_data[0]
    print(f"System: {example['messages'][0]['content']}")
    print(f"\nCustomer: {example['messages'][1]['content']}")
    print(f"\nAgent: {example['messages'][2]['content']}")

    input("\n\nPress Enter to see the comparison demo...")

    # Run the main comparison
    run_comparison_demo()

    # Explain the difference
    explain_the_difference()

    # Show use cases
    demo_different_use_cases()

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)