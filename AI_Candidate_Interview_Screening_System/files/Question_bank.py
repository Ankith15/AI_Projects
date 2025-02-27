from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Set up Google API key (ensure it's set correctly)
load_dotenv()
# Function to generate the next question based on the user's last response
def generate_next_question(last_response):
    prompt_template = """
    You are an HR interviewing a candidate for an AI-related position.
    Based on the candidate's last response, ask the next relevant interview question.
    Start with an introduction and adapt your questions based on previous answers.

    Last response: {last_response}

    Next question:
    """
    
    # Set up the language model (Google Generative AI)
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["last_response"])
    question_chain = LLMChain(prompt=prompt, llm=model)
    
    # Generate the next question
    next_question = question_chain.run({"last_response": last_response})
    
    # Debug log to see the generated question
    print(f"Generated next question: {next_question}")
    
    return next_question

# Function to score the user's answer (out of 10)
def evaluate_answer(answer):
    score = len(answer.split()) // 5  # Simple logic: 1 point for every 5 words
    return min(score, 10)

# Function to handle the interview flow
def handle_interview(user_answer, last_question, total_score, question_count):
    if question_count >= 4:
        return None, total_score  # Stop after 4 questions

    # Score the user's answer
    score = evaluate_answer(user_answer)
    total_score += score

    # Generate the next question based on the user's answer
    next_question = generate_next_question(user_answer)  # New question based on last answer
    
    return next_question, total_score

# Initialization function to start the interview
def start_interview():
    initial_question = "Please introduce yourself."
    return initial_question, 0, 0  # Start with 0 score and 0 questions asked
