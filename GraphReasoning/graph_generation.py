from GraphReasoning.graph_tools import *
from GraphReasoning.utils import *
from GraphReasoning.graph_analysis import *

import copy

import re
from IPython.display import display, Markdown

import markdown2
import pdfkit

 
import uuid
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import networkx as nx
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random
from pyvis.network import Network

from tqdm.notebook import tqdm

import itertools
import seaborn as sns
palette = "hls"

import uuid
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns  # For more attractive plotting

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 
import pandas as pd

import transformers
from transformers import logging

logging.set_verbosity_error()

import re
from IPython.display import display, Markdown

import markdown2
import pdfkit

 
import uuid
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import networkx as nx
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random
from pyvis.network import Network

from tqdm.notebook import tqdm

import seaborn as sns
palette = "hls"

import uuid
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns  # For more attractive plotting

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Code based on: https://github.com/rahulnyk/knowledge_graph

def extract (string, start='[', end=']'):
    start_index = string.find(start)
    end_index = string.rfind(end)
     
    return string[start_index :end_index+1]
def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk,
           # **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

def concepts2Df(concepts_list) -> pd.DataFrame:
    ## Remove all NaN entities
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()
    )

    return concepts_dataframe


def df2Graph(dataframe: pd.DataFrame, generate, repeat_refine=0, verbatim=False,
          
            ) -> list:
  
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, generate, {"chunk_id": row.chunk_id}, repeat_refine=repeat_refine,
                                verbatim=verbatim,#model
                               ), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list


def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: str(x).lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: str(x).lower())

    return graph_dataframe

import sys
from yachalk import chalk
sys.path.append("..")

import json

def graphPrompt(input: str, generate, metadata={}, #model="mistral-openorca:latest",
                repeat_refine=0,verbatim=False,
               ):
    
    SYS_PROMPT_GRAPHMAKER = (
        "You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context, including well-defined and widely used names of materials, systems, methods. \n\n"
        "Format your output as a list of JSON. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
        "   }, {...}\n"
        "]"
        ""
        "Examples:"
        "Context: ```Alice is Marc's mother.```\n"
        "[\n"
        "   {\n"
        '       "node_1": "Alice",\n'
        '       "node_2": "Marc",\n'
        '       "edge": "is mother of"\n'
        "   }, "
        "{...}\n"
        "]"
        "Context: ```Silk is a strong natural fiber used to catch prey in a web. Beta-sheets control its strength.```\n"
        "[\n"
        "   {\n"
        '       "node_1": "silk",\n'
        '       "node_2": "fiber",\n'
        '       "edge": "is"\n'
        "   }," 
        "   {\n"
        '       "node_1": "beta-sheets",\n'
        '       "node_2": "strength",\n'
        '       "edge": "control"\n'
        "   },"        
        "   {\n"
        '       "node_1": "silk",\n'
        '       "node_2": "prey",\n'
        '       "edge": "catches"\n'
        "   },"
        "{...}\n"
        "]\n\n"
        "Analyze the text carefully and produce around 10 triplets, making sure they reflect consistent ontologies.\n"
        )
        
    USER_PROMPT = f"Context: ```{input}``` \n\nOutput: "
    
    print (".", end ="")
    response  =  generate( system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=USER_PROMPT)
    if verbatim:
        print ("---------------------\nFirst result: ", response)
   
    SYS_PROMPT_FORMAT = ('You respond in this format:'
                 '[\n'
                    "   {\n"
                    '       "node_1": "A concept from extracted ontology",\n'
                    '       "node_2": "A related concept from extracted ontology",\n'
                    '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
                    '   }, {...} ]\n'  )    
    USER_PROMPT = (f'Read this context: ```{input}```.'
                  f'Read this ontology: ```{response}```'
                 f'\n\nImprove the ontology by renaming nodes so that they have consistent labels that are widely used in the field of materials science.'''
                 '')
    response  =  generate( system_prompt=SYS_PROMPT_FORMAT,
                          prompt=USER_PROMPT)
    if verbatim:
        print ("---------------------\nAfter improve: ", response)
    
    USER_PROMPT = f"Context: ```{response}``` \n\n Fix to make sure it is proper format. "
    response  =  generate( system_prompt=SYS_PROMPT_FORMAT, prompt=USER_PROMPT)
    response =   response.replace ('\\', '' )
    if verbatim:
        print ("---------------------\nAfter clean: ", response)
    
    if repeat_refine>0:
        for rep in tqdm(range (repeat_refine)):
            

            
            USER_PROMPT = (f'Insert new triplets into the original ontology. Read this context: ```{input}```.'
                          f'Read this ontology: ```{response}```'
                          f'\n\nInsert additional triplets to the original list, in the same JSON format. Repeat original AND new triplets.\n'
                         '') 
            response  =  generate( system_prompt=SYS_PROMPT_GRAPHMAKER, 
                                  prompt=USER_PROMPT)
            if verbatim:
                print ("---------------------\nAfter adding triplets: ", response)
            USER_PROMPT = f"Context: ```{response}``` \n\n Fix to make sure it is proper format. "
            response  =  generate( system_prompt=SYS_PROMPT_FORMAT, prompt=USER_PROMPT)
            response =   response.replace ('\\', '' )
            USER_PROMPT = (f'Read this context: ```{input}```.'
                          f'Read this ontology: ```{response}```'
                         f'\n\nRevise the ontology by renaming nodes and edges so that they have consistent and concise labels.'''
                        
                         '') 
            response  =  generate( system_prompt=SYS_PROMPT_FORMAT,  
                                  prompt=USER_PROMPT)            
            if verbatim:
                print (f"---------------------\nAfter refine {rep}/{repeat_refine}: ", response)

     
    USER_PROMPT = f"Context: ```{response}``` \n\n Fix to make sure it is proper format. "
    response  =  generate( system_prompt=SYS_PROMPT_FORMAT, prompt=USER_PROMPT)
    response =   response.replace ('\\', '' )
    
    try:
        response=extract (response)
       
    except:
        print (end='')
    
    try:
        result = json.loads(response)
        print (result)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result

def colors2Community(communities) -> pd.DataFrame:
    
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    ## Melt the dataframe into a list of nodes
    df['node_1'] = df['node_1'].astype(str)
    df['node_2'] = df['node_2'].astype(str)
    df['edge'] = df['edge'].astype(str)
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2
    
def make_graph_from_text (txt,generate,
                          include_contextual_proximity=False,
                          graph_root='graph_root',
                          chunk_size=2500,chunk_overlap=0,
                          repeat_refine=0,verbatim=False,
                          data_dir='./data_output_KG/',
                          save_PDF=False,#TO DO
                          save_HTML=True,
                         ):    
    
    ## data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)     
     
    outputdirectory = Path(f"./{data_dir}/") #where graphs are stored from graph2df function
    
 
    splitter = RecursiveCharacterTextSplitter(
        #chunk_size=5000, #1500,
        chunk_size=chunk_size, #1500,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    pages = splitter.split_text(txt)
    print("Number of chunks = ", len(pages))
    if verbatim:
        display(Markdown (pages[0]) )
    
    df = documents2Dataframe(pages)

    ## To regenerate the graph with LLM, set this to True
    regenerate = True
    
    if regenerate:
        concepts_list = df2Graph(df,generate,repeat_refine=repeat_refine,verbatim=verbatim) #model='zephyr:latest' )
        dfg1 = graph2Df(concepts_list)
        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory)
        
        dfg1.to_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|", index=False)
        df.to_csv(outputdirectory/f"{graph_root}_chunks.csv", sep="|", index=False)
        dfg1.to_csv(outputdirectory/f"{graph_root}_graph_clean.csv", #sep="|", index=False
                   )
        df.to_csv(outputdirectory/f"{graph_root}_chunks_clean.csv", #sep="|", index=False
                 )
    else:
        dfg1 = pd.read_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|")
    
    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = 4 
      
    if verbatim:
        print("Shape of graph DataFrame: ", dfg1.shape)
    dfg1.head()### 
    
    if include_contextual_proximity:
        dfg2 = contextual_proximity(dfg1)
        dfg = pd.concat([dfg1, dfg2], axis=0)
        #dfg2.tail()
    else:
        dfg=dfg1
        
    
    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
        .reset_index()
    )
    #dfg
        
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    print ("Nodes shape: ", nodes.shape)
    
    G = nx.Graph()
    node_list=[]
    node_1_list=[]
    node_2_list=[]
    title_list=[]
    weight_list=[]
    chunk_id_list=[]
    
    ## Add nodes to the graph
    for node in nodes:
        G.add_node(
            str(node)
        )
        node_list.append (node)
    
    ## Add edges to the graph
    for _, row in dfg.iterrows():
        
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            weight=row['count']/4
        )
        
        node_1_list.append (row["node_1"])
        node_2_list.append (row["node_2"])
        title_list.append (row["edge"])
        weight_list.append (row['count']/4)
         
        chunk_id_list.append (row['chunk_id'] )

    try:
            
        df_nodes = pd.DataFrame({"nodes": node_list} )    
        df_nodes.to_csv(f'{data_dir}/{graph_root}_nodes.csv')
        df_nodes.to_json(f'{data_dir}/{graph_root}_nodes.json')
        
        df_edges = pd.DataFrame({"node_1": node_1_list, "node_2": node_2_list,"edge_list": title_list, "weight_list": weight_list } )    
        df_edges.to_csv(f'{data_dir}/{graph_root}_edges.csv')
        df_edges.to_json(f'{data_dir}/{graph_root}_edges.json')
        
    except:
        
        print ("Error saving CSV/JSON files.")
    
    communities_generator = nx.community.girvan_newman(G)
    #top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    
    if verbatim:
        print("Number of Communities = ", len(communities))
        
    if verbatim:
        print("Communities: ", communities)
    
    colors = colors2Community(communities)
    if verbatim:
        print ("Colors: ", colors)
    
    for index, row in colors.iterrows():
        G.nodes[row['node']]['group'] = row['group']
        G.nodes[row['node']]['color'] = row['color']
        G.nodes[row['node']]['size'] = G.degree[row['node']]
            
    net = Network(
             
            notebook=True,
         
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            
            filter_menu=False,
        )
        
    net.from_nx(G)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
   
    net.show_buttons()
    
    graph_HTML= f'{data_dir}/{graph_root}_grapHTML.html'
    graph_GraphML=  f'{data_dir}/{graph_root}_graphML.graphml'  #  f'{data_dir}/resulting_graph.graphml',
    nx.write_graphml(G, graph_GraphML)
    
    if save_HTML:
        net.show(graph_HTML,
            )

    if save_PDF:
        output_pdf=f'{data_dir}/{graph_root}_PDF.pdf'
        pdfkit.from_file(graph_HTML,  output_pdf)
    else:
        output_pdf=None
    res_stat=graph_statistics_and_plots_for_large_graphs(G, data_dir=data_dir,include_centrality=False,
                                                       make_graph_plot=False,)
        
    print ("Graph statistics: ", res_stat)
    return graph_HTML, graph_GraphML, G, net, output_pdf

import time
from copy import deepcopy

def add_new_subgraph_from_text(txt,generate,node_embeddings,tokenizer, model,
                               original_graph_path_and_fname,
                               data_dir_output='./data_temp/', verbatim=True,
                               size_threshold=10,chunk_size=10000,
                               do_Louvain_on_new_graph=True,include_contextual_proximity=False,repeat_refine=0,similarity_threshold=0.95, do_simplify_graph=True,#whether or not to simplify, uses similiraty_threshold defined above
                               return_only_giant_component=False,
                               save_common_graph=False,G_to_add=None,graph_GraphML_to_add=None,
                              ):

    display (Markdown(txt[:256]+"...."))
    graph_GraphML=None
     
    G_new=None
    res=None
    assert not (G_to_add is not None and graph_GraphML_to_add is not None), "G_to_add and graph_GraphML_to_add cannot be used together. Pick one or the other to provide a graph to be added."
 
    try:
        start_time = time.time() 
        idx=0
        
        if verbatim:
            print ("Now create or load new graph...")

        if graph_GraphML_to_add==None and G_newlymade==None: #make new if no existing one provided
            print ("Make new graph from text...")
            _, graph_GraphML_to_add, G_to_add, _, _ =make_graph_from_text (txt,generate,
                                      include_contextual_proximity=include_contextual_proximity,
                                      
                                     data_dir=data_dir_output,
                                     graph_root=f'graph_new_{idx}',
                                    
                                        chunk_size=chunk_size,   repeat_refine=repeat_refine, 
                                      verbatim=verbatim,
                                       
                                  )
            if verbatim:
                print ("Generated new graph from text provided: ", graph_GraphML_to_add)

        else:
            if verbatim:
                print ("Instead of generating graph, loading it or using provided graph...(any txt data provided will be ignored...)")

            if graph_GraphML_to_add!=None:
                print ("Loading graph: ", graph_GraphML_to_add)
        
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        print ("ALERT: Graph generation failed...for idx=",idx)
    
    print ("Now add node to existing graph...")
    
    try:
        #Load original graph
        G = nx.read_graphml(original_graph_path_and_fname)
        
        if G_to_add!=None:
            G_loaded=H = deepcopy(G_to_add)
            if verbatim:
                print ("Using provided graph to add (any txt data provided will be ignored...)")
        else:
            if verbatim:
                print ("Loading graph to be added either newly generated or provided.")
            G_loaded = nx.read_graphml(graph_GraphML_to_add)
        
        res_newgraph=graph_statistics_and_plots_for_large_graphs(G_loaded, data_dir=data_dir_output,include_centrality=False,
                                                       make_graph_plot=False,root='new_graph')
        print (res_newgraph)
        
        G_new = nx.compose(G,G_loaded)

        if save_common_graph:
            print ("Identify common nodes and save...")
            try:
                
                common_nodes = set(G.nodes()).intersection(set(G_loaded.nodes()))
    
                subgraph = G_new.subgraph(common_nodes)
                graph_GraphML=  f'{data_dir_output}/{graph_root}_common_nodes_before_simple.graphml' 
                nx.write_graphml(subgraph, graph_GraphML)
            except: 
                print ("Common nodes identification failed.")
            print ("Done!")
        
        if verbatim:
            print ("Now update node embeddings")
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model)
        print ("Done update node embeddings.")
        if do_simplify_graph:
            if verbatim:
                print ("Now simplify graph.")
            G_new, node_embeddings =simplify_graph (G_new, node_embeddings, tokenizer, model , 
                                                    similarity_threshold=similarity_threshold, use_llm=False, data_dir_output=data_dir_output,
                                    verbatim=verbatim,)
            if verbatim:
                print ("Done simplify graph.")
            
        if verbatim:
            print ("Done update graph")
        
        if size_threshold >0:
            if verbatim:
                print ("Remove small fragments")            
            G_new=remove_small_fragents (G_new, size_threshold=size_threshold)
            node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)
        
        if return_only_giant_component:
            if verbatim:
                print ("Select only giant component...")   
            connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
            G_new = G_new.subgraph(connected_components[0]).copy()
            node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)
            
        print (".")
        if do_Louvain_on_new_graph:
            G_new=graph_Louvain (G_new, 
                      graph_GraphML=None)
            if verbatim:
                print ("Don Louvain...")

        print (".")
         
        graph_root=f'graph'
        graph_GraphML=  f'{data_dir_output}/{graph_root}_augmented_graphML_integrated.graphml'  #  f'{data_dir}/resulting_graph.graphml',
        print (".")
        nx.write_graphml(G_new, graph_GraphML)
        print ("Done...written: ", graph_GraphML)
        res=graph_statistics_and_plots_for_large_graphs(G_new, data_dir=data_dir_output,include_centrality=False,
                                                       make_graph_plot=False,root='assembled')
        
        print ("Graph statistics: ", res)

    except:
        print ("Error adding new graph.")
        print (end="")

    return graph_GraphML, G_new, G_loaded, G, node_embeddings, res
