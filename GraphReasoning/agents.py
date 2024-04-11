import transformers
from transformers import logging
from GraphReasoning.utils import *


#transformers.logging.set_verbosity_info()
logging.set_verbosity_error()

from guidance import models, gen, select

from guidance.models import LlamaCpp
from guidance.models import Chat
import re
from guidance import gen, select, system, user, assistant,  newline
from IPython.display import display, Markdown

import markdown2
import pdfkit

# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# Ah, me hearty matey! But yer question be a puzzler! A human cannot eat a helicopter in one sitting, as helicopters are not edible. They be made of metal, plastic, and other materials, not food!

class ZephyrLlamaCppChat(LlamaCpp, Chat):
    def get_role_start(self, role_name, **kwargs):
        if role_name == "user":
            return "<|user|>\n"
        
        elif role_name == "assistant":
            return "<|assistant|>\n"
        
        elif role_name == "system":
            return "<|system|>\n"
            
    def get_role_end(self, role_name=None):
        if role_name == "user":
            return "</s>"
        elif role_name == "assistant":
            return "</s>"
        elif role_name == "system":
            return "</s>"

class ConversationAgent:
    def __init__(self, chat_model,   name: str, instructions: str, context_turns: int = 2,
                temperature=0.1,  ):
        self._chat_model = chat_model
        self._name = name
        self._instructions = instructions
        self._my_turns = []
        self._interlocutor_turns = []
        self._went_first = False
        self._context_turns = context_turns
     #   self._q=q
        self.temperature=temperature

    @property
    def name(self) -> str:
        return self._name
    
    def get_conv(self, ) -> str:

        return self._my_turns
        
    def reply(self, interlocutor_reply: str | None = None) -> str:
        if interlocutor_reply is None:
            self._my_turns = []
            self._interlocutor_turns = []
            self._went_first = True
        else:
            self._interlocutor_turns.append(interlocutor_reply)

        # Get trimmed history

        print 
        my_hist = self._my_turns[(1-self._context_turns):]
        interlocutor_hist = self._interlocutor_turns[-self._context_turns:]

        # Set up the system prompt
        curr_model = self._chat_model
        with system():
            #curr_model += f"Your name is {self.name}. {self._instructions}"
            curr_model += f"{self._instructions}"
            '''
            if len(interlocutor_hist) == 0:
                curr_model += "Introduce yourself"
            elif len(interlocutor_hist) == 1:
                curr_model += "Introduce yourself before continuing the conversation"

            '''
        # Replay the last few turns
        for i in range(len(my_hist)):
            with user():
                curr_model += interlocutor_hist[i]
            with assistant():
                curr_model += my_hist[i]

        if len(interlocutor_hist) > 0:
            with user():
                curr_model += interlocutor_hist[-1]
        #else:
        #    with user():
        #        if q != '':
        #            curr_model += f"Answer this question: {self._q}"

        #print ("CURRENT: ",curr_model)
        with assistant():
            curr_model += gen(name='response', max_tokens=1024,temperature= self.temperature)

        self._my_turns.append(curr_model['response'])
        return curr_model['response']

def conversation_simulator(
    bot0: ConversationAgent,
    question_gpt,
    question_gpt_name='Engineer',answer_gpt_name='Biologist', 
    question_temperature = 0.7,
    question_asker_instructions='You ALWAYS ask tough questions. ',
    q='What is bioinspiration?',
    total_turns: int = 5,data_dir='./',
    marker_ch='>>> ',start_with_q=False,only_last=True,
    )-> list[dict[str,str]]:

    conversation_turns = []
    last_reply = q 
    first=True
    for _ in range(total_turns):

        last_reply = bot0.reply(last_reply)
        
        conversation_turns.append(dict(name=bot0.name, text=last_reply))

        if first:
            first=False
            if only_last:
                
                txt= f'Consider this question and response.\n\n### Question: {q}\n\n### Response: {last_reply}'

            
            else:
                conv=get_entire_conversation(q, conversation_turns,marker_ch=marker_ch,start_with_q=start_with_q, question_gpt_name=question_gpt_name)
                txt=f'### Consider this conversation between {question_gpt_name} and {answer_gpt_name}:\n\n{conv}\n\n"'
                

        else:
            
            if only_last:
                txt= f'Consider this question and response.\n\n### Question: {q}\n\n### Response: {last_reply}'

            else:
                conv=get_entire_conversation(q, conversation_turns,marker_ch=marker_ch,start_with_q=start_with_q,question_gpt_name=question_gpt_name)
                txt=f'### Consider this conversation between {question_gpt_name} and {answer_gpt_name}:\n\n{conv}\n\n"'
        
             
        with system():
            lm = question_gpt+question_asker_instructions+"\n\nYou MUST respond with ONE new probing question. " 

        with user():
            lm += f"""{txt}\n\n### Instruction: Respond with a SINGLE follow-up question that critically challenges the response. 
DO NOT answer the question or comment on it yet. 
\n\nThe single question is: """

        with assistant():
            q_new = lm+gen(name='question', temperature=question_temperature, 
                           max_tokens=500, stop=newline)

        q_new=q_new['question'].replace('"', '')
        
        last_reply=q_new
        
        conversation_turns.append(dict(name=question_gpt_name, text=last_reply))

         
    return conversation_turns


def read_and_summarize(gpt, txt='This is a conversation.', q='',
                      ):
    
    with system():
        lm = gpt + "You analyze text and provide an accurate account of the content from all sides discussed."
    
    with user():        
        lm += f"""Carefully read this conversation: 

<<<{txt}>>>
Accurately summarize the conversation and identify the key points made.

Think step by step: 
""" 
        
    with assistant():        
        lm+=gen('summary', max_tokens=1024)
    
    with user():        
        lm += f'Now list the salient insights as bullet points.'
        
    with assistant():        
        lm+=gen('bullet', max_tokens=1024)
     
    with user():        
        lm += f'Identify the single most important takeaway in the conversation and how it answers the original question, <<<{q}>>>.'
        
    with assistant():        
        lm+=gen('takeaway', max_tokens=1024)

    return lm['summary'], lm['bullet'], lm['takeaway']
    

def answer_question (gpt_question_asker, gpt, q='I have identified this amino acid sequence: AAAAAIIAAAA. How can I use it? ',
                     
                    bot_name_1="Biologist",
                    bot_instructions_1 = f"""You are a biologist. You are taking part in a discussion, from a life science perspective.
Keep your answers brief, but accurate, and creative.
""",
                    bot_name_2="Engineer",
                    bot_instructions_2 = """You are a critical engineer. You are taking part in a discussion, from the perspective of engineering.
Keep your answers brief, and always challenge statements in a provokative way. As a creative individual, you inject ideas from other fields. """,
                    question_temperature = 0.1,
                     conv_temperature=0.3,
                     
                    total_turns=4,
                    delete_last_question=True, #whether or not the last question is deleted (since it is not actually answered anyway)
                    save_PDF=True,
                    PDF_name=None,  save_dir='./', 
                     txt_file_path=None, marker_ch='>>> ',start_with_q=False,only_last=True,
                     
                    ):
    
    bot_1 = ConversationAgent(chat_model=gpt,  name=bot_name_1, instructions=bot_instructions_1, 
                              context_turns=total_turns, temperature=conv_temperature)
    
    conversation_turns = conversation_simulator(bot_1, question_gpt=gpt_question_asker, 
                                                question_gpt_name=bot_name_2,answer_gpt_name=bot_name_1,
                                                question_temperature=question_temperature,
                                                question_asker_instructions=bot_instructions_2,
                                                q=q, 
                                                total_turns=total_turns, data_dir=save_dir,marker_ch=marker_ch,start_with_q=start_with_q,only_last=only_last)

    if delete_last_question:
        conversation_turns.pop()
        
    txt=''
    txt+=f"The question discussed is: **{q.strip()}**\n\n"
     
    print ("-----------------------------------------")
    for turn in conversation_turns:
        
        txt +=f"**{turn['name'].strip ()}**: {turn['text']}\n\n"

    summary, bullet, keytakaway = read_and_summarize(gpt ,txt, q=q)

    integrated = f"""#### Question and conversation:
    
{txt} 

#### Summary:

{summary}

#### List of key points:

{bullet}

#### Key takeaway:

**{keytakaway.strip()}**
"""

    if save_PDF:
        # Convert Markdown to HTML
        html_text = markdown2.markdown(integrated)
        
        # Convert HTML to PDF and save it
        max_len_fname=64
        if PDF_name==None:
            PDF_name=f'{save_dir}{q[:max_len_fname].strip()}.pdf'
        
        pdfkit.from_string(html_text, PDF_name)
    
    max_len_fname=64
    if txt_file_path==None:
        txt_file_path = f'{save_dir}{q[:max_len_fname].strip()}.txt'
    save_raw_txt=remove_markdown_symbols(integrated)
    
    with open(txt_file_path, 'w') as file:
        file.write(save_raw_txt)

    return conversation_turns, txt, summary, bullet, keytakaway, integrated, save_raw_txt


def get_entire_conversation (q, conversation_turns, marker_ch='### ', start_with_q=False, question_gpt_name='Question: '):
    txt=''
    
    if start_with_q:
        txt+=f"{marker_ch}The question discussed is: {q.strip()}\n\n"
    else:
        txt+=f"{marker_ch}{question_gpt_name}: {q.strip()}\n\n"
    
    print ("-----------------------------------------")
    for turn in conversation_turns:
        
        txt +=f"{marker_ch}{turn['name'].strip ()}: {turn['text']}\n\n"
    return txt


#############################################################################################################################
######################################################### LLamaIndex based ################################################## 
#############################################################################################################################

from llama_index.core.memory import ChatMemoryBuffer
 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.chat_engine import SimpleChatEngine

def get_chat_engine_from_index_LlamaIndex(llm,index, chat_token_limit=2500,verbose=False, chat_mode="context",
                               system_prompt='You are a chatbot, able to have normal interactions, as well as talk about context provided.'):
    memory = ChatMemoryBuffer.from_defaults(token_limit=chat_token_limit)
    
    chat_engine = index.as_chat_engine(llm=llm,
        chat_mode=chat_mode,
        memory=memory,
        system_prompt=system_prompt,verbose=verbose,
    )

    return chat_engine
    
def get_answer_LlamaIndex (llm, #model, tokenizer, 
                           q, system_prompt="You are an expert in materials science.", chat_engine=None,
                        max_new_tokens=1024, #temperature=0.7, 
                           messages_to_prompt=None,chat_token_limit=2500,chat_mode="context",
                completion_to_prompt=None,index=None, verbose=False):
    
    if chat_engine==None:
        
        if index != None:
           
            chat_engine=get_chat_engine_from_index_LlamaIndex(llm,index, chat_token_limit=chat_token_limit,verbose=verbose,chat_mode=chat_mode,
                                   system_prompt=f'You are a chatbot, able to have normal interactions, as well as talk about data provided. {system_prompt}')
        else:
            chat_engine = SimpleChatEngine.from_defaults(llm=llm, system_prompt=system_prompt)
            
    response = chat_engine.stream_chat(q)
    for token in response.response_gen:
        print(token, end="")
    return response.response, chat_engine



class ConversationAgent_LlamaIndex:
    def __init__(self, llm,
                
                 name: str, instructions: str,# context_turns: int = 2,
              
                 index=None,chat_token_limit=2500,verbose=False,chat_mode="context",
                ):
      
        self._name = name
        self._instructions = instructions
        self._source_nodes =[]
       
        if index != None:
            print (f"Set up chat engine, with index, verbose={verbose}, chat_mode={chat_mode}.")
            
            self.chat_engine=get_chat_engine_from_index_LlamaIndex(llm,index, chat_token_limit=chat_token_limit,verbose=verbose,chat_mode=chat_mode,
                       system_prompt=f'You are a chatbot, able to have normal interactions, as well as talk about data provided.\n\n{self._instructions}')
        else:
            self.chat_engine = SimpleChatEngine.from_defaults(llm=llm, system_prompt=self._instructions)
   

    @property
    def name(self) -> str:
        return self._name
    
    def get_conv(self, ) -> str:

        return self.chat_engine.chat_history
    def get_source_nodes(self, ) -> str:

        return self._source_nodes

    def reset_chat(self, ):
        self.chat_engine.reset()
        
    def reply(self, question) -> str:
        response = self.chat_engine.stream_chat(question   )
        for token in response.response_gen:
            print(token, end="")

        self._source_nodes.append (response.source_nodes)
            
        return response.response, response

def conversation_simulator_LlamaIndex(
      
     llm_answer, llm_question,
    question_gpt_name='Engineer',answer_gpt_name='Biologist', answer_instructions='You answer correctly.',
  
    question_asker_instructions='You always respond with a single, tough, question. ',
    q='What is bioinspiration?',
    total_turns: int = 5,data_dir='./',
    marker_ch='>>> ',start_with_q=False,only_last=True, 
    marker_ch_outer='### ',sample_question='',
     
    answer_index=None,question_index=None, verbose=False,chat_mode="context",chat_token_limit=2500,
    iterate_on_question=False,#whether to revise question after initial draft, 
    include_N_turns_in_question_development=9999,single_shot_question=True,
    iterate_on_question_with_earlier_context=False, #whether or not to iterate on question with all earlier context of just the question draft
    )-> list[dict[str,str]]:

    answer_agent = ConversationAgent_LlamaIndex(llm_answer,
                                              
                                                name=answer_gpt_name, instructions=answer_instructions, 
                         
                                                index=answer_index,verbose=verbose,chat_mode=chat_mode,chat_token_limit=chat_token_limit,
                                               )
    
    conversation_turns = []
    q_new = q #None

    conversation_turns.append(dict(name=question_gpt_name, text=q_new))

    
    print (f"### {question_gpt_name}: {q}\n")
    for _ in range(total_turns):

        print (f"### {answer_gpt_name}: ", end="")
        
        last_reply, response = answer_agent.reply(q_new)
        
        
        conversation_turns.append(dict(name=answer_gpt_name, text=last_reply))

        if only_last:
                
            txt= f'Consider this question and response.\n\n{marker_ch_outer}Question: {q}\n\n{marker_ch_outer} Response: {last_reply}'

        else:
            NN=include_N_turns_in_question_development
            
            NN = NN + 1 if NN % 2 else NN  # Adjust NN to be even if it's not
            conv=get_entire_conversation_LlamaIndex(q, conversation_turns[-NN:],marker_ch=marker_ch,start_with_q=start_with_q, question_gpt_name=question_gpt_name)
            
            txt=f'{marker_ch_outer}Read this conversation between {question_gpt_name} and {answer_gpt_name}:\n\n```{conv}```\n\n"'

        if single_shot_question: # SINGLE SHOT QUESTION 
    
            q=f"""{txt}\n\n{marker_ch_outer}Instruction: Respond with a SINGLE follow-up question that critically challenges the earlier responses. 
    
DO NOT answer the question or comment on it yet. Do NOT repeat a question that was asked in the earlier conversation.{sample_question}
    
The single question is:"""
    
            q=f"""{txt}\n\n{marker_ch_outer}Please generate a thoughtful and challenging follow-up question. {sample_question}{question_gpt_name}:"""
            
            print (f"\n\n### {question_gpt_name}: ", end="")
            
            q_new, q_chat=get_answer_LlamaIndex (llm_question,#model, tokenizer, 
                                            q=q, #temperature=question_temperature,
                    
                                            index=question_index,verbose=verbose,chat_mode=chat_mode,chat_token_limit=chat_token_limit,
                 system_prompt=question_asker_instructions+"You MUST respond with ONE new probing question. ONLY provide the question.")

        else:  # MULTI SHOT QUESTION
            q=f"""{txt}\n\n{marker_ch_outer}Instruction: Summarize the conversation, with details. Include logic and reasoning, and think step by step."""
            print (f"\n\n### {question_gpt_name}, summary: ", end="")
            summary_for_q, chat_engine=get_answer_LlamaIndex (llm_question, q=q,  #messages_to_prompt=messages_to_prompt,
               
                                 system_prompt="You analyze text and develop questions.", chat_engine=None)
            q=f"""{marker_ch_outer}Please generate a thoughtful and challenging follow-up question. {sample_question}\n\nThe question is:"""
            print (f"\n\n### {question_gpt_name}: ", end="")
            q_new, chat_engine=get_answer_LlamaIndex (llm_question, q=q,  #messages_to_prompt=messages_to_prompt,
                
                                 system_prompt="You analyze text and develop questions.",chat_engine=chat_engine)
        
                
        if iterate_on_question:
            if iterate_on_question_with_earlier_context==False:
                q_chat=None #start with new chat
            print (f"\n\n### {question_gpt_name} (iterate): ", end="")
            q_new, _=get_answer_LlamaIndex (llm_question,#model, tokenizer, 
                                            q=f"Make sure >>>{q_new}<<< is a SINGLE question.\n\nDO NOT answer the question. If it is a single question, just reply with the question.{sample_question}\n\nThe SINGLE question is: ", #temperature=question_temperature,
                                         
                                            index=question_index,verbose=verbose,chat_mode=chat_mode,chat_token_limit=chat_token_limit,
             system_prompt="You pose questions.",chat_engine=q_chat
                                                )

        q_new=q_new.replace('"', '')
        
        print (f"\n")
        
        conversation_turns.append(dict(name=question_gpt_name, text=q_new))

         
    return conversation_turns, answer_agent.get_conv(), response, answer_agent


def read_and_summarize_LlamaIndex( llm, txt='This is a conversation.', q='',
                   ):
    q=f"""Carefully read this conversation: 

>>>{txt}<<<

Accurately summarize the conversation and identify the key points made.

Think step by step: 
""" 

    summary, chat_engine=get_answer_LlamaIndex (llm, q=q,   
                                 system_prompt="You analyze text and provide an accurate account of the content from all sides discussed.")

    q=f'Now list the salient insights as bullet points.'
    
    bullet, chat_engine=get_answer_LlamaIndex (llm, q=q, 
                                 system_prompt="You analyze text and provide an accurate account of the content from all sides discussed.",
                                              chat_engine=chat_engine)


    q=f'Identify the single most important takeaway in the conversation and how it answers the original question, <<<{q}>>>.'
    takeaway, chat_engine=get_answer_LlamaIndex (llm, q=q, 
                                 system_prompt="You analyze text and provide an accurate account of the content from all sides discussed.",
                                                chat_engine=chat_engine)

     
    return summary, bullet, takeaway

def answer_question_LlamaIndex ( #model, tokenizer, 
                                
                      llm_answer,
                    llm_question,   llm_summarize, 
    q='I have identified this amino acid sequence: AAAAAIIAAAA. How can I use it? ',
                    bot_name_1="Biologist",
                    bot_instructions_1 = f"""You are a biologist. You are taking part in a discussion, from a life science perspective.
Keep your answers brief, but accurate, and creative.
""",
                    bot_name_2="Engineer",
                    bot_instructions_2 = """You are a critical engineer. You are taking part in a discussion, from the perspective of engineering.
Keep your answers brief, and always challenge statements in a provokative way. As a creative individual, you inject ideas from other fields. """,
                 
                     include_N_turns_in_question_development=99999,
                    total_turns=4,
                    delete_last_question=True, #whether or not the last question is deleted (since it is not actually answered anyway)
                    save_PDF=True,sample_question='',
                    PDF_name=None,  save_dir='./', 
                     txt_file_path=None, marker_ch='>>> ',marker_ch_outer='### ',
                     start_with_q=False,only_last=True,single_shot_question=True,
                      messages_to_prompt=None,question_index=None, answer_index=None,chat_mode="context",chat_token_limit=2500,
                completion_to_prompt=None,iterate_on_question=False,iterate_on_question_with_earlier_context=True,verbose=False,
                    ):
    
    conversation_turns, answer_agent_conv, response, answer_agent = conversation_simulator_LlamaIndex( llm_answer, llm_question,#  model, tokenizer, 
                                                question_gpt_name=bot_name_2,answer_gpt_name=bot_name_1,
                                               question_asker_instructions=bot_instructions_2,
                                                q=q, question_index=question_index, answer_index=answer_index,
  include_N_turns_in_question_development=include_N_turns_in_question_development, 
    single_shot_question=single_shot_question,                                                                
                                                total_turns=total_turns, data_dir=save_dir,marker_ch=marker_ch,marker_ch_outer=marker_ch_outer,
                                                           start_with_q=start_with_q,only_last=only_last, sample_question=sample_question,
                                                      verbose=verbose,chat_mode=chat_mode,chat_token_limit=chat_token_limit,
                                                    iterate_on_question=iterate_on_question,iterate_on_question_with_earlier_context=iterate_on_question_with_earlier_context,
                              )

    if delete_last_question:
        conversation_turns.pop()
        
    txt=''
    txt+=f"The question discussed is: **{q.strip()}**\n\n"
     
    print ("-----------------------------------------")
    for turn in conversation_turns:
        
        txt +=f"**{turn['name'].strip ()}**: {turn['text']}\n\n"

    summary, bullet, keytakaway = read_and_summarize_LlamaIndex(llm_summarize,#model, tokenizer ,
                                                                txt, q=q,  )

    integrated = f"""#### Question and conversation:
    
{txt} 

#### Summary:

{summary}

#### List of key points:

{bullet}

#### Key takeaway:

**{keytakaway.strip()}**
"""

    if save_PDF:
        # Convert Markdown to HTML
        html_text = markdown2.markdown(integrated)
        
        # Convert HTML to PDF and save it
        max_len_fname=64
        if PDF_name==None:
            PDF_name=f'{save_dir}{q[:max_len_fname].strip()}.pdf'
        
        pdfkit.from_string(html_text, PDF_name)
    
    max_len_fname=64
    if txt_file_path==None:
        txt_file_path = f'{save_dir}{q[:max_len_fname].strip()}.txt'
    save_raw_txt=remove_markdown_symbols(integrated)
    
    with open(txt_file_path, 'w') as file:
        file.write(save_raw_txt)

    return conversation_turns, txt, summary, bullet, keytakaway, integrated, save_raw_txt, answer_agent_conv, response, answer_agent


def get_entire_conversation_LlamaIndex (q, conversation_turns, marker_ch='### ', start_with_q=False, question_gpt_name='Question: '):
    txt=''
    
    if start_with_q:
        txt+=f"{marker_ch}The question discussed is: {q.strip()}\n\n"
    else:
        txt=''
    
    #print ("-----------------------------------------")
    for turn in conversation_turns:
        
        txt +=f"{marker_ch}{turn['name'].strip ()}: {turn['text']}\n\n"
    return txt.strip()
