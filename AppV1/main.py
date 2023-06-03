import streamlit as st
from dotenv import load_dotenv
import openai
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from prompts import snoop_rap_template, snoop_title_template, snoop_imgprompt_template


def generate_image(image_description):
    img_response = openai.Image.create(
        prompt = image_description,
        n=1,
        size="512x512")
    img_url = img_response['data'][0]['url']
    return img_url


def main():
    st.title("Snoop Dogg AI Rap Bot")
    st.subheader("Powered by OpenAI, LangChain, Streamlit")
    user_input = st.text_input("Enter Prompt:")

    if st.button("Generate") and user_input:
        with st.spinner('Generating...'):
            rapper_template = PromptTemplate(
                input_variables=['user_input'],
                template=snoop_rap_template
            )
            title_template = PromptTemplate(
                input_variables=['rap'],
                template=snoop_title_template
            )
            imgprompt_template = PromptTemplate(
                input_variables=['input'],
                template=snoop_imgprompt_template
            )
            
            llm = OpenAI(temperature=1)

            rap_chain = LLMChain(llm=llm, prompt=rapper_template, verbose=True)
            title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
            imgprompt_chain = LLMChain(llm=llm, prompt=imgprompt_template, verbose=True)

            rap_response = rap_chain.run(user_input)
            title_response = title_chain.run(rap_response)

            gen_input = "Snoop dogg styled poster for title " + title_response
            generated_img = generate_image(gen_input)
            st.image(generated_img)
            st.subheader(title_response)
            st.write(rap_response)


if __name__ == '__main__':
    load_dotenv()
    main()

