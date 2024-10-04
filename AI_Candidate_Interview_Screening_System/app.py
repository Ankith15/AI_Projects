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

    if 'total_score' not in st.session_state:
        st.session_state.current_question, st.session_state.total_score = start_interview()

    # Display the current question
    st.write(st.session_state.current_question)

    # Get user's audio response
    user_answer = get_audio()
    st.write(f"Your answer: {user_answer}")

    # Get the next question based on the user's answer
    if st.button("Submit Answer"):
        next_question, score, st.session_state.total_score = handle_interview(user_answer, st.session_state.current_question, st.session_state.total_score)
        st.write(f"Score for this answer: {score}/10")
        st.session_state.current_question = next_question

        # Display the next question or overall rank after 4 questions
        if st.session_state.total_score >= 40:  # Assuming 10 points max for each question
            st.write(f"Overall Rank: {st.session_state.total_score}/40")
        else:
            st.write(st.session_state.current_question)

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
