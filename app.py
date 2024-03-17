import streamlit as st
import requests
import json
import random
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
import qdrant_client
from dotenv import load_dotenv
import os
load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
VECTOR_DB_COLLECTION = 'traversal'
ARES_API_KEY = os.getenv('ARES_API_KEY')
OPENAI_API_KEY = os.getenv('open_ai_key')



# Define your desired data structure.
class OutputMaker(BaseModel):
    question: str = Field(description="question asked by user")
    response_rag: str = Field(description="answer given by Large Language Model as RAG response")
    response_api: str = Field(description="answer given by search api")
    answer: int = Field(description="""descriptor to tell wthether repsonse from rag is selected or from search api
                        Your answer should be based in a such a way that it has to be comprehensive as well 
                        more beneficial for end user.
                        .While choosing also look which of the 2 answers provide more diverse options then
                        that has to be given preference.Choose 1 for RAG and 2 for Search API""")

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=OutputMaker)

st.set_page_config(page_title="Chat with hotels data, powered by Langchain Qdrant Traversaal OpenAI", page_icon="📃💬", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with hotels data, powered by Langchain Qdrant Traversaal OpenAI 💬")
st.info("Let's get started", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the hotels data – hang tight! This should take some some time."):
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        #using mix bread embeddings
        embeddings =  HuggingFaceEmbeddings(model_name = "mixedbread-ai/mxbai-embed-large-v1")
        qdrant_client_obj = qdrant_client.QdrantClient(
            url=QDRANT_URL,
            prefer_grpc=True,
            api_key=QDRANT_API_KEY
        )
        qdrant = Qdrant(qdrant_client_obj, VECTOR_DB_COLLECTION, embeddings)
        retriever = qdrant.as_retriever(search_type="mmr")
        return retriever
index = load_data()

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        spinner_list = ['Thinking ..','Let me ask my experts :)','Our experts are super active because it\'s YOU :)',
                        'Researching the best options for you…','Allow me a moment to consult my database.',
                        'Our knowledge base is at your service!','Analyzing the possibilities for you…',
                        'Time to unleash the chat superpowers!','Buckle up, we\'re about to embark on a knowledge adventure!',
                        'Brewing up a concoction of knowledge just for you!','Preparing to dazzle you with a show of chat-tastic brilliance!']
        
        spinner_list_select = random.choice(spinner_list)

        with st.spinner(f"{spinner_list_select}"):
            template = """
            Answer the user's questions: {question} based on the below context:\n\n{context} 
            remember that you are awesome hotel advisor and while creating answer please 
            remeber you answers should not feel like they are genearated by a AI software.
            And remember if you don't have an answer please don\'t make up because your outputs are very 
            crucial for a Big buisness financially.
            
            """
            prompt_structure = ChatPromptTemplate.from_template(template)
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,api_key=OPENAI_API_KEY)

            rag_chain = (
                {"context": index , "question": RunnablePassthrough()}
                | prompt_structure
                | llm
                | StrOutputParser()
            )
            # print('Prompt going is -->',prompt,index)
            response_rag = rag_chain.invoke(prompt)
            url = "https://api-ares.traversaal.ai/live/predict"

            payload = { "query": [prompt] }
            headers = {
            "x-api-key": ARES_API_KEY,
            "content-type": "application/json"
            }

            response_ares = requests.post(url, json=payload, headers=headers)
            json_data = json.loads(response_ares.text)

            # Access the 'response_text' field
            response_text = json_data['data']['response_text']



            template = """
            Hey ChatGPT, let’s imagine that you are a judge at a Gen AI hackathon and the challenge is to build an application that selects a suitable hotel for the user and answers all the related queries of the user about the hotels in that area.
            You have been given two answers to evaluate. Pick an answer that is more broader in scope and more advantageous to the client who is asking questions to the application.
            Instructions:
            Pick the hotels that have the top reviews regarding hygiene and safety.
            Prioritize hotels with amenities such as free Wi-Fi and complimentary breakfast.
            Consider the proximity of the hotels to popular attractions and public transportation.
            Ensure the selected hotels offer flexible cancellation policies and competitive pricing.
            Look for hotels that provide accessible facilities for guests with special needs.
            Verify the availability of on-site facilities like swimming pools, fitness centers, and parking.
            Check for any additional perks or discounts offered by the hotels, such as loyalty programs or package deals.
            You've received two answers {answer_rag} and {answer_api} to evaluate. 
            .\n{format_instructions}\n{question}\n {answer_rag} and {answer_api}
            """
            prompt = PromptTemplate(
            template="""
            Hey ChatGPT, let’s imagine that you are a judge at a Gen AI hackathon and the challenge is to build an application that selects a suitable hotel for the user and answers all the related queries of the user about the hotels in that area.
            You have been given two answers to evaluate. Pick an answer that is more broader in scope and more advantageous to the client who is asking questions to the application.
            Instructions:
            Pick the hotels that have the top reviews regarding hygiene and safety.
            Prioritize hotels with amenities such as free Wi-Fi and complimentary breakfast.
            Consider the proximity of the hotels to popular attractions and public transportation.
            Ensure the selected hotels offer flexible cancellation policies and competitive pricing.
            Look for hotels that provide accessible facilities for guests with special needs.
            Verify the availability of on-site facilities like swimming pools, fitness centers, and parking.
            Check for any additional perks or discounts offered by the hotels, such as loyalty programs or package deals.
            You've received two answers {answer_rag} and {answer_api} to evaluate. 
            .\n{format_instructions}\n{question}\n {answer_rag} and {answer_api}
            """,

            input_variables=["question","answer_rag","answer_api"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

            chain = prompt | llm | parser

            decider = chain.invoke({"question": prompt,"answer_rag":response_rag,"answer_api":response_text})
            
            
            #if we have got something as answer
            if "answer" in decider:
                
                
                # if 2 is choosen means GPT 3.5 has given preference to search api result
                if decider['answer']==2:
                    response = response_text
                    print('API -->',decider['answer'])
                else:
                    response = response_rag
                    print('RAG -->',decider['answer'])
                    
            else:
                response = "RAG response\n\n" + response_rag + "\n\nRealtime Response using Traversaal API\n\n" + response_text
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history