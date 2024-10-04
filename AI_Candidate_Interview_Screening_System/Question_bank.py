from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Function to generate the next question based on the user's last response
def generate_next_question(last_response):
    prompt_template = """
    You are an HR interviewing a candidate for an AI-related position.
    Based on the candidate's last response, ask the next relevant interview question.
    You must always start with an introduction question, and then adapt your questions based on the user's previous answers.

    Last response: {last_response}

    Next question:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["last_response"])
    question_chain = LLMChain(prompt=prompt, llm=model)
    
    next_question = question_chain.run({"last_response": last_response})
    return next_question

# Function to score the user's answer (out of 10)
def evaluate_answer(answer):
    score = len(answer.split()) // 5  # Simple logic, adjust based on your requirements
    return min(score, 10)

# Function to handle the interview flow
def handle_interview(user_answer, last_question, total_score):
    score = evaluate_answer(user_answer)
    total_score += score

    next_question = generate_next_question(user_answer)
    return next_question, score, total_score

# Initialization function to start the interview
def start_interview():
    initial_question = "Please introduce yourself."
    return initial_question, 0  # Start with a score of 0
