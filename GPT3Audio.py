#!/usr/bin/env python
# coding: utf-8
!pip install openai


# In[3]:


import os
import openai
import json
import streamlit as st
import streamlit.components.v1 as components
import requests
import numpy as np 

from io import BytesIO
from time import sleep


# In[4]:


assembly_auth_key = "ae51526c6848401d8b5599d77ceb696d"


# In[5]:


openai.api_key = "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"


# In[6]:


headers = {
    'authorization': assembly_auth_key, 
    'content-type': 'application/json',
}

upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcription_endpoint = "https://api.assemblyai.com/v2/transcript"


# In[7]:


def upload_to_assemblyai(file_path):

    def read_audio(file_path):

        with open(file_path, 'rb') as f:
            while True:
                data = f.read(5_242_880)
                if not data:
                    break
                yield data

    upload_response =  requests.post(upload_endpoint, 
                                     headers=headers, 
                                     data=read_audio(file_path))

    return upload_response.json().get('upload_url')


# In[8]:


def transcribe(upload_url): 

    json = {"audio_url": upload_url}
    
    response = requests.post(transcription_endpoint, json=json, headers=headers)
    transcription_id = response.json()['id']

    return transcription_id


# In[9]:


def get_transcription_result(transcription_id): 

    current_status = "queued"

    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcription_id}"

    while current_status not in ("completed", "error"):
        
        response = requests.get(endpoint, headers=headers)
        current_status = response.json()['status']
        
        if current_status in ("completed", "error"):
            return response.json()['text']
        else:
            sleep(10)


# In[10]:


def call_gpt3(prompt):

    response = openai.Completion.create(engine = "text-davinci-001", 
                                        prompt = prompt, max_tokens = 50)
    return response["choices"][0]["text"]


# In[11]:


def main():

    st.title("Talking to GPT-3")
    file_path = "input.wav"

    record_audio(file_path)

    upload_url = upload_to_assemblyai(file_path)
    st.write('Prompt uploaded to AssemblyAI')

    transcription_id = transcribe(upload_url)
    st.write('Prompt Sent for Transciption to AssemblyAI')

    prompt = get_transcription_result(transcription_id)

    st.write('Prompt Transcribed...Sending to GPT-3')
    st.info(prompt)

    gpt_output = call_gpt3(prompt)

    st.write('Response Received from GPT-3')
    st.success(gpt_output)


# In[ ]:




