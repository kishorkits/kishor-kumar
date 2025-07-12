import streamlit as st

st.title("📄 Display Text File Content")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    st.text_area("File Content", content, height=300)
