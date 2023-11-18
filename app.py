import streamlit as st
import os
import openai
import requests
import re
import time


from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from dataclasses import dataclass
from typing import Literal


# Identifies between User and AI messages
@dataclass
class Message:
    origin: Literal["User", "AI"]
    message: str


# Store chat history between User & AI
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []


initialize_session_state()


def extract_text(url):
    try:
        # Send an HTTP request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text from the parsed HTML
            text = soup.get_text()
            title = soup.title.get_text()

            # Excludes special characters from string to avoid errors when creating files
            title = re.sub('[^A-Za-z0-9]+', ' ', title)

            with open(f'data/{title}.txt', 'w', encoding='utf-8') as content:
                content.write(text)
            st.success("Successfully Retrieved!")
            return text
    except:
        st.error(
            f"Failed to retrieve page.")
        return None


# Extract text from a YouTube video given its URL
def extract_video_text(url):
    try:
        if 'v=' in url:
            # Extract the video identifier from the URL after v=""
            url_ID = url.split('v=')[1].split('&')[0]
            source = YouTubeTranscriptApi.get_transcript(url_ID)

            text = ""
            for line in source:
                text += " " + line['text']

            # Fetch the video title to name the text file
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            title = soup.title.get_text()

            # Removes special characters as to avoid errors in file creation
            title = re.sub('[^A-Za-z0-9]+', ' ', title)

            with open(f'data/{title}.txt', 'w', encoding='utf-8') as content:
                content.write(text)
            st.success("Successfully Retrieved!")
            return text
        else:
            st.error("Invalid URL")
    except:
        st.error("Unsupported Video")


def read_data():
    text = ""
    if os.path.isdir("data") is False:
        os.mkdir("data")
    data_dir = os.path.join(os.getcwd(), "data")
    for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            text += f.read()
    return text


def create_docs(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_text(text)
    return docs


def create_embedding(docs):
    embeddings = OpenAIEmbeddings()
    doc_search = FAISS.from_texts(docs, embeddings)
    return doc_search


def response_chain(doc_search, prompt, LLM):
    from langchain.chains.question_answering import load_qa_chain

    chain = load_qa_chain(llm=LLM, chain_type="stuff")
    docs = doc_search.similarity_search(prompt)
    response = chain.run(input_documents=docs, question=prompt)
    return response


def isValid(apiKey):
    try:
        openai.api_key = apiKey
        openai.Completion.create(
            engine="text-davinci-003",
            prompt=".",
            max_tokens=5,
            temperature=0.8,
        )
    except:
        return False
    else:
        return True


def main():
    st.set_page_config(
        page_title="SumupAI",
        page_icon="📝"
    )

    st.title("SumupAI")
    st.write("Engage in dynamic conversations with your favorite YouTube videos, podcasts, and articles.")

    OPENAI_API_KEY = st.text_input(
        'OpenAI API Key', placeholder='sk-...IAzF', type="password")

    st.divider()

    if OPENAI_API_KEY:

        LLM = ChatOpenAI(
            temperature=0.8,
            model_name="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY
        )

        source = st.radio("Select Input Type",
                          ("Article", "Video"))

        if source == 'Video':
            user_input = st.text_input("Insert Video URL", "")
            source_button = st.button("Submit")
            if source_button and user_input != "":
                extract_video_text(user_input)
        elif source == 'Article':
            user_input = st.text_input("Insert Article URL", "")
            source_button = st.button("Submit")
            if source_button and user_input != "":
                extract_text(user_input)

        st.divider()

        prompt_placehoolder = st.form("chat-form")
        with prompt_placehoolder:
            st.markdown("**CHAT** - Enter to Submit")
            cols = st.columns((6, 1))
            question = cols[0].text_input(
                "Chat",
                label_visibility="collapsed"
            )
            prompt_button = cols[1].form_submit_button(
                "Submit",
                type="primary"
            )

            try:
                if prompt_button and question != "":
                    with st.spinner("Processing..."):
                        response = response_chain(create_embedding(
                            create_docs(read_data())), prompt=question, LLM=LLM)
                        st.session_state.history.append(
                            Message("User", question))
                        st.session_state.history.append(
                            Message("AI", response))
                        for chat in st.session_state.history:
                            st.write(f"**{chat.origin}**: {chat.message}")

            except:
                if isValid(OPENAI_API_KEY) is False:
                    st.error("Invalid API Key", icon="🚨")
                else:
                    error_placeholder = st.empty()
                    seconds = 20
                    for i in range(seconds, 0, -1):
                        error_placeholder.error(
                            f"Rate Limit Exceeded. Please Wait {i} Seconds")
                        time.sleep(1)
                    error_placeholder.empty()


if __name__ == "__main__":
    main()
