from dotenv import load_dotenv, dotenv_values
from openai import OpenAI, OpenAIError
from pathlib import Path
import os


SYSTEM_PROMPT = """ 
        You are a clinical researcher that reads medical charts and answers questions about them. 
        Use only the information in the text provided to answer the question.
        If a patient denies something, do not include it in your answer.
        After you provide an answer, you immediately stop talking
 """
    
config = {}
USER_PROMPT = ""


DEFAULT_MODEL = 'openai/gpt-oss-120b'
def load_config():
    """Loads config file from the environment"""
    config.update({**dotenv_values(".env")})


def create_client(**kwargs):
    print('Config')
    client = OpenAI(
        api_key=config['API_KEY'],
        base_url=config['BASE_URL'],
        **kwargs)
    return client

def create_query(messages:str,model=None, reasoning_effort='high',temperature=0.5,seed=0,client=None)->tuple:


    if client is None:
        client = create_client()

    if type(messages) is not str:
        messages = USER_PROMPT

    
    stream = client.chat.completions.create(
        model=config['model'] if model is None else model,
        messages=[
            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },
            {
                "role":"user",
                "content":messages
            }
        ],
                reasoning_effort=reasoning_effort,
        temperature=temperature,
        seed=seed,
        stream=True,
        )

    response_chunks = []
    reasoning_chunks = []
    for chunk in stream:
        if chunk.choices:
            # --- Extract reasoning
            if getattr(chunk.choices[0].delta, 'reasoning_content', None) is not None:
                reasoning_chunks.append(chunk.choices[0].delta.reasoning_content)

            # --- Extract response
            if getattr(chunk.choices[0].delta, 'content', None) is not None:
                response_chunks.append(chunk.choices[0].delta.content)
        
    return ''.join(response_chunks), ''.join(reasoning_chunks)

def read_query(query):
    print(query)



def main():
    load_config()
    message = USER_PROMPT
    query,reasoning = create_query(message)
    read_query(query)


if __name__ == '__main__':
    main()