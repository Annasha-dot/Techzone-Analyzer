import streamlit as st
# Import the function to analyze the TZ_analyzer
from TZ_analyser import analyse_query


# Define the Streamlit app
st.title("Smart TechZone Analyzer")

TZ_link = st.text_input("Enter the link to the TechZone received")
TZ_description = st.text_area("Enter the description of the TechZone")

problem_type = st.selectbox("Problem Type",
                            ["----Select the Type of Problem----", 
                             "Crash", 
                             "Timeout issues", 
                             "High CPU usage", 
                             "Memory leaks", 
                             "Network issues", 
                             "Ping failures", 
                             "Script failures",
                             "Others"])

component = st.selectbox("Component",
                         ["----Select the Component----", 
                          "Management component", 
                          "Execution component",
                          "Data component",
                          "Static component",
                          "Protocol component",
                          "Others"])

# Streamlit button to trigger the RAG process
if st.button("Analyze TechZone"):
    response = analyse_query(
       TZ_link, TZ_description, problem_type, component
    )
    st.subheader("Analysis Result")
    st.json(response)