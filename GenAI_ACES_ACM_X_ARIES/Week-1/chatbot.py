import pathlib
import textwrap

import google.generativeai as genai

# Used to securely store your API key
from google.colab import userdata

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

genai.configure(api_key =GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash')

# response = model.generate_content("Translate I am a beautiful pig to Russian")

# to_markdown(response.text)

# import PIL.Image

# img = PIL.Image.open('image.jpg')
# response = model.generate_content(["Write a short poem based on this picture", img], stream = True)
# response.resolve()
# to_markdown(response.text)

import os
!pip install groq
from groq import Client

GROQ_API_KEY = userdata.get('GROQ')

groq = Client(api_key = GROQ_API_KEY,)

def call_groq(prompt, context):

  past_context = "".join(context)

  chat_completion = groq.chat.completions.create(messages =[{"role":"user", "content": "Keeping in mind this context of the past 5 or less messages: "+ past_context + "(END OF CONTEXT) reply to the following new prompt, do not quote this instruction or say anything that would tell the user this command has been given, the prompt beings after ':' right here: "+ prompt,}], model = "meta-llama/llama-4-scout-17b-16e-instruct")

  return chat_completion.choices[0].message.content

def call_gemini(prompt, context, response):
  past_context = "".join(context)
  return (model.generate_content("Keeping in mind this context (START OF CONTEXT)"+past_context+"(END OF CONTEXT) for this prompt (START OF PROMPT)"+ prompt+" (END OF PROMPT) another LLM generated this response(START OF RESPONSE)"+response+"(END OF RESPONSE). Give a critical review of this prompt with improvements, and/or highlight corrections for factual inaccuracies in an actionable manner for the model along with the prompt itself").text)

def run_chat_agent():
    """
    Simulates a chatbot agent using a while loop and LLM API calls.

    Your implementation should:
    - Continuously prompt the user for input.
    - Use either Groq or Gemini API to respond.
    - Allow the user to type 'exit' or 'quit' to end the conversation.

    Optional:
    - Add validation before making the API call.
    - Simulate multiple agents by calling different LLMs with distinct system prompts.
    - Route the user query through two agents (e.g., responder and critic).

    Hint:
    Use `input()` to capture user queries, and wrap your API calls inside functions like:
    `call_groq(prompt)` or `call_gemini(prompt, system_prompt=None)`
    """
    pass  # TODO: Implement your conversational loop here

    context = []
    n = 5
    print("Ask a question, request something, simply talk or type 'exit' or 'quit' to quit!")
    
    while True:
      print("continue:")
      prompt = input()
      if len(context) > n:
        del context[0]
      print("[USER]: " + prompt)

      if prompt == 'exit' or prompt == 'quit':
        print("Goodbye!")
        break
      else:
        groq_response = call_groq(prompt, context)
        context.append(prompt)
        gemini_response = call_gemini(prompt, context, groq_response)
        print("[REFINED GROQ]: "+ call_groq(gemini_response, context))
        

run_chat_agent()