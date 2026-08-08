import sys
import streamlit as st
from streamlit.web import cli as stcli

def draw ():
    st.title("Streamlit with Spyder")
    user_name = st.text_input("What is your name?", "Developer")
    status_message = f"Hello {user_name}, welcome to your Streamlit dashboard!"
    st.write(status_message)

def initApp ():
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())

if __name__ == "__main__":
    is_running = st.runtime.exists()
    draw() if is_running else initApp()