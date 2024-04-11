from openai import OpenAI
import base64
import requests
from datetime import datetime
from GraphReasoning.graph_tools import *

from GraphReasoning.utils import *
from GraphReasoning.graph_analysis import *

import openai

def generate_OpenAIGPT ( system_prompt='You are a materials scientist.', prompt="Decsribe the best options to design abrasive materials.",
              temperature=0.2,max_tokens=2048,timeout=120,
             
             frequency_penalty=0, 
             presence_penalty=0, 
             top_p=1.,  
               openai_api_key='',gpt_model='gpt-4-vision-preview', organization='',
             ):
    client = openai.OpenAI(api_key=openai_api_key,
                      organization =organization)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        model=gpt_model,
        timeout=timeout,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
    )
    return chat_completion.choices[0].message.content


def reason_over_image_OpenAI (system_prompt='You are a scientist.', prompt='Carefully analyze this graph. Be creative and synthesize new research ideas to build sustainable mycelium materials.',
                      image_path='IMAGES/H1000_E_bridggingcentrality_alt_2.png',
                       temperature=0.2,max_tokens=2048,timeout=120,
                         #gpt_model='gpt-3.5-turbo',
                         frequency_penalty=0, 
                         presence_penalty=0, openai_api_key='',gpt_model='gpt-4-vision-preview', organization='',
                         top_p=1., #local_llm=None,
                              verbatim=False,
                                  ):

    if verbatim:
        print ("Prompt: ", prompt)
    def encode_image(image_path):
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Getting the base64 string
    base64_image = encode_image(image_path)
    
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {openai_api_key}"
    }
    
    payload = {
      "model": gpt_model,
      "messages": [
          {
                "role": "system",
                "content": system_prompt,
            },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      
        "max_tokens":max_tokens,
     
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload,)
   
    if verbatim:
        display (Markdown(response.json()['choices'][0]['message']['content']))

    return response.json()['choices'][0]['message']['content']



def reason_over_image_and_graph_via_triples (path_graph, generate, image_path='',
                          keyword_1 = "music and sound",
                         # local_llm=None,
                          keyword_2 = "apples",include_keywords_as_nodes=True,
                          inst_prepend='',
                          
                          instruction = 'Now, reason over them and propose a research hypothesis.',
                          verbatim=False,
                          N_limit=None,temperature=0.3,
                          keywords_separator=' --> ',system_prompt='You are a scientist who uses logic and reasoning.',
                          max_tokens=4096,prepend='You are given a set of information from a graph that describes the relationship between materials, structure, properties, and properties. You analyze these logically through reasoning.\n\n',
                          
                          save_files=True,data_dir='./',visualize_paths_as_graph=True, display_graph=True,words_per_line=2,
                         ):
    
    print ("Reason over graph and image: ", image_path)
    
    make_dir_if_needed(data_dir)
    task=inst_prepend+''
    

    join_strings = lambda strings: '\n'.join(strings)
    join_strings_newline = lambda strings: '\n'.join(strings)

    node_list=print_node_pairs_edge_title(path_graph)
    if N_limit != None:
        node_list=node_list[:N_limit]

    if verbatim:
        print ("Node list: ", node_list)
        
     
    if include_keywords_as_nodes:
        task=task+f"The following is a graph provided from an analysis of relationships between the concepts of {keyword_1} and {keyword_2}.\n\n"
    task=task+f"Consider this list of nodes and relations in a knowledge graph:\n\nFormat: node_1, relationship, node_2\n\nThe data is:\n\n{join_strings_newline( node_list)}\n\n"

    task=task+f"{instruction}"
     
    if verbatim:
        print ( "###############################################################\nTASK:\n", task)
    
 
    response=generate(system_prompt=system_prompt,  
         prompt=task, max_tokens=max_tokens, temperature=temperature,image_path=image_path,)
    
    if verbatim:
        display(Markdown("**Response:** "+response ))

    return response ,  path_graph,  fname, graph_GraphML
    

from openai import OpenAI  # OpenAI Python library to make API calls
import requests  # used to download images
import os  # used to access filepaths
from PIL import Image  # used to print and edit images
import base64
from IPython.display import display, Image
import json

def develop_prompt_from_text_and_generate_image (response, generate_OpenAIGPT, image_dir_name='./image_temp/', number_imgs=1,
                                                size="1024x1024",show_img=True,max_tokens=2048,temperature=0.3,
                                                 quality='hd', style='vivid', direct_prompt=None,  openai_api_key='',
                                                 gpt_model='gpt-4-0125-preview', organization='', dalle_model="dall-e-3",
                                                 system_prompt="You make prompts for DALLE-3."
                                                ):

    
    image_dir = os.path.join(os.curdir, image_dir_name)
    make_dir_if_needed(image_dir)
    img_list=[]
    if direct_prompt == None:
        task=f'''Consider this description of a novel material: {response}

Develop a well-constructed, detailed and clear prompt for DALLE-3 that allows me to visualize the new material design. 
        
The prompt should be written such that the resulting image presents a clear reflection of the material's real microstructure and key features. Make sure that the resulting image does NOT include any text.
'''
        
        response=generate_OpenAIGPT(system_prompt=system_prompt, #local_llm=local_llm,
                 prompt=task, max_tokens=max_tokens, temperature=temperature, )
        display (Markdown("Image gen prompt:\n\n"+response))
    else:
        response=direct_prompt
        display (Markdown("Image gen prompt already provided:\n\n"+response))
     
    # set a directory to save DALL·E images to

    client = openai.OpenAI(api_key=openai_api_key,
                      organization =organization)
    generation_response = client.images.generate(
        model = dalle_model,
        prompt=response,
        n=number_imgs,
        style=style,
        quality=quality,
        size=size,
        
        response_format="b64_json",
    )
    

    for index, image_dict in enumerate(generation_response.data):
        image_data = base64.b64decode(image_dict.b64_json)
        
        # Get the current time
        time_part = datetime.now().strftime("%Y%m%d_%H%M%S")

        image_file = os.path.join(image_dir_name, f"generated_image_{time_part}_{response[:32]}_{index}.png")
        with open(image_file, mode="wb") as png:
            png.write(image_data)
        display(Image(data=image_data))
     
    return img_list