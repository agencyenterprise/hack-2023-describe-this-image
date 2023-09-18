from langchain.llms import OpenAI, Replicate
import streamlit as st
from langchain.prompts import PromptTemplate

import os

openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_llm = OpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
replicate_llm = Replicate(model="stability-ai/sdxl:2b017d9b67edd2ee1401238df49d75da53c523f36e363881e057f5dc3ed3c5b2")


def generate_image_description():
  prompt = 'create a prompt to generate an image that represents something that can be described in one sentence. For example, "A group of friends laughing and posing for a selfie with colorful balloons in a sunlit park, celebrating a special occasion" or "A young girl sitting under a tree, engrossed in a book with a radiant smile on her face while imagining herself as the adventurous protagonist"'
  return openai_llm(prompt)


def generate_image(prompt: str):

    return replicate_llm(prompt)


def analyze_user_description(original_description: str, user_description: str):
  prompt = PromptTemplate.from_template("""
  Given the following image description: "{original_description}"

  Give a 0 to 100 score to how the following user description describes the same image. Also, give tips on language,
  spelling and grammar improvements. Use only the language used by the user's description in your response.

  User description: "{user_description}".
  """)
  return openai_llm(prompt.format(original_description=original_description, user_description=user_description))


def refresh():
  st.session_state.generated_description = generate_image_description()
  st.session_state.generated_image = generate_image(st.session_state.generated_description)


st.set_page_config(page_title="Describe This Image", page_icon="🖼️")
st.title('🖼️ Describe this image')

st.markdown("""
This app uses AI to generate an image and a description of that image.

**Your goal is to describe the image in one sentence, using any language as you want. Be as creative and descriptive as possible.**

The AI will then score your description and give you tips on how to improve it, also considering language, grammar and spelling.
""")


with st.spinner('Generating an image for you...'):
  if 'generated_description' not in st.session_state or 'generated_image' not in st.session_state:
    refresh()
  # st.text(st.session_state.generated_description)
  # st.text(st.session_state.generated_image)
  st.image(st.session_state.generated_image)

submitted = False
with st.form('my_form'):
  text = st.text_area('Your description')
  submitted = st.form_submit_button('Submit')

if submitted:
  with st.spinner('Analyzing your description...'):
    st.info(analyze_user_description(original_description=st.session_state.generated_description, user_description=text))
    st.button("Start over", on_click=refresh)
