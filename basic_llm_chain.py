"""
Basic LangChain LLM Chain Implementation
This script demonstrates the fundamental concepts of LangChain including:
- Setting up an LLM (Groq)
- Creating prompt templates
- Building LLM chains
- Using output parsers
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Verify API key is loaded
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables. Please create a .env file with your API key.")


def example_1_basic_llm():
    """
    Example 1: Basic LLM usage
    Demonstrates how to initialize and use a basic LLM
    """
    print("\n=== EXAMPLE 1: Basic LLM Usage ===")
    
    # Initialize the LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    # Simple invocation
    response = llm.invoke("What is LangChain?")
    print(f"Response: {response.content}\n")


def example_2_prompt_template():
    """
    Example 2: Using Prompt Templates
    Shows how to create reusable prompt templates
    """
    print("\n=== EXAMPLE 2: Prompt Templates ===")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains technical concepts in simple terms."),
        ("user", "Explain {topic} in 2-3 sentences.")
    ])
    
    # Format the prompt
    formatted_prompt = prompt.invoke({"topic": "Retrieval-Augmented Generation"})
    print(f"Formatted Prompt: {formatted_prompt}\n")
    
    # Get response
    response = llm.invoke(formatted_prompt)
    print(f"Response: {response.content}\n")


def example_3_llm_chain():
    """
    Example 3: Complete LLM Chain
    Demonstrates chaining components together using LCEL (LangChain Expression Language)
    """
    print("\n=== EXAMPLE 3: LLM Chain with LCEL ===")
    
    # Initialize components
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a coding assistant that provides concise code examples."),
        ("user", "Show me a {language} code example for {task}")
    ])
    
    output_parser = StrOutputParser()
    
    # Create the chain using LCEL (pipe operator)
    chain = prompt | llm | output_parser
    
    # Invoke the chain
    result = chain.invoke({
        "language": "Python",
        "task": "reading a CSV file with pandas"
    })
    
    print(f"Chain Result:\n{result}\n")


def example_4_multiple_queries():
    """
    Example 4: Batch Processing
    Shows how to process multiple queries efficiently
    """
    print("\n=== EXAMPLE 4: Batch Processing ===")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template(
        "What is the capital of {country}?"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # Batch invoke
    countries = [
        {"country": "France"},
        {"country": "Japan"},
        {"country": "Brazil"}
    ]
    
    results = chain.batch(countries)
    
    for i, result in enumerate(results):
        print(f"{countries[i]['country']}: {result}")
    
    print()


def example_5_streaming():
    """
    Example 5: Streaming Responses
    Demonstrates how to stream LLM responses
    """
    print("\n=== EXAMPLE 5: Streaming Responses ===")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template(
        "Write a short poem about {subject}"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    print("Streaming response for 'artificial intelligence':")
    for chunk in chain.stream({"subject": "artificial intelligence"}):
        print(chunk, end="", flush=True)
    
    print("\n")


def main():
    """
    Main function to run all examples
    """
    print("=" * 60)
    print("LangChain LLM Chain - Basic Implementation Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        example_1_basic_llm()
        example_2_prompt_template()
        example_3_llm_chain()
        example_4_multiple_queries()
        example_5_streaming()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please make sure:")
        print("1. You have created a .env file with your GROQ_API_KEY")
        print("2. All dependencies are installed (pip install -r requirements.txt)")


if __name__ == "__main__":
    main()
