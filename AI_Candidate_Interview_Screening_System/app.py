import streamlit as st
from Audio_Screening import get_audio
from Question_bank import start_interview, handle_interview


# Home page to select the interview role
def home_page():
    st.title("Welcome To AI Interview")
    st.write('Please select the role you are interested in')

    if st.button('Data Science'):
        st.session_state.page = "Data Science"
    elif st.button("Machine Learning"):
        st.session_state.page = "Machine Learning"
    elif st.button("AI Engineer"):
        st.session_state.page = "AI Engineer"


# Data Science interview page
def data_science_page():
    st.title("Data Science Interview Questions")
    st.write('Please answer the questions carefully.')

    # Initialize session state variables if not already set
    if 'total_score' not in st.session_state:
        st.session_state.current_question, st.session_state.total_score, st.session_state.question_count = start_interview()

    # Display the current question
    st.write(f"Question: {st.session_state.current_question}")

    # Get user's audio response
    user_answer = get_audio()
    st.write(f"Your answer: {user_answer}")

    # Button to submit the answer
    if st.button("Submit Answer"):
        # Check if fewer than 4 questions have been asked
        if st.session_state.question_count < 4:
            # Call the handle_interview function to process the answer and fetch the next question
            next_question, new_total_score = handle_interview(
                user_answer, 
                st.session_state.current_question, 
                st.session_state.total_score, 
                st.session_state.question_count
            )

            # Update session state
            st.session_state.total_score = new_total_score
            st.session_state.question_count += 1

            if next_question is not None and next_question != st.session_state.current_question:
                # Update to the next question
                st.session_state.current_question = next_question
                st.write(f"Next Question: {st.session_state.current_question}")
            else:
                # If no more questions, end the interview
                st.write(f"Interview Completed! Your total score: {st.session_state.total_score}/40")

        else:
            # End interview after 4 questions
            st.write(f"Interview Completed! Your total score: {st.session_state.total_score}/40")

    # Option to go back to home
    if st.button("Back"):
        st.session_state.page = "Home"



# Machine Learning interview page
def machine_learning_page():
    st.title("Machine Learning Interview Questions")
    st.write("This is the Machine Learning question-answer system.")

    if st.button("Back"):
        st.session_state.page = "Home"


# AI Engineer interview page
def ai_engineer_page():
    st.title("AI Engineer Interview Questions")
    st.write("This is the AI Engineer question-answer system.")

    if st.button("Back"):
        st.session_state.page = 'Home'


# Main function to handle page navigation
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Data Science":
        data_science_page()
    elif st.session_state.page == "Machine Learning":
        machine_learning_page()
    elif st.session_state.page == "AI Engineer":
        ai_engineer_page()


if __name__ == "__main__":
    main()
