from GraphReasoning.graph_tools import *
from GraphReasoning.utils import *
from GraphReasoning.graph_generation import *
import networkx as nx
import matplotlib.pyplot as plt
import os

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


import networkx as nx
import random
import numpy as np
from copy import deepcopy
import numpy as np
import random
from datetime import datetime

import random
import numpy as np
from copy import deepcopy
import numpy as np
import random
from datetime import datetime


def euclidean_distance(vec1, vec2):
    """Calculate the Euclidean distance between two vectors."""
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def heuristic_path_with_embeddings(G, embedding_tokenizer, embedding_model, source, target, node_embeddings, top_k=3, second_hop=False,
                                   data_dir='./', save_files=True, verbatim=False,):
    G = deepcopy(G)

    if verbatim:
        print ("Original: ", source, "-->", target)
    source=find_best_fitting_node_list(source, node_embeddings  , embedding_tokenizer, embedding_model, 5)[0][0].strip()
    target=find_best_fitting_node_list(target, node_embeddings  , embedding_tokenizer, embedding_model, 5)[0][0].strip()

    #if verbatim:
    print ("Selected: ", source, "-->", target)
    
    def heuristic(current, target):
        """Estimate distance from current to target using embeddings."""
        return euclidean_distance(node_embeddings[current], node_embeddings[target])

    def sample_path(current, visited):
        path = [current]
        while current != target:
            neighbors = [(neighbor, heuristic(neighbor, target)) for neighbor in G.neighbors(current) if neighbor not in visited]
            if not neighbors:
                # Dead end reached, backtrack if possible
                if len(path) > 1:
                    visited.add(path.pop())  # Mark the dead-end node as visited and remove it from the path
                    current = path[-1]  # Backtrack to the previous node
                    continue
                else:
                    # No path found
                    return None
            else:
                neighbors.sort(key=lambda x: x[1])
                top_neighbors = neighbors[:top_k] if len(neighbors) > top_k else neighbors
                next_node = random.choice(top_neighbors)[0]
    
                path.append(next_node)
                visited.add(next_node)  # Mark the node as visited
                current = next_node
    
                if len(path) > 2 * len(G):  # Prevent infinite loops

                    print (f"No path found between {source} and {target}")
                    return None 
        return path

    visited = set([source])  # Initialize visited nodes set
    path = sample_path(source, visited)
    if path is None:
        print (f"No path found between {source} and {target}")
        return None, None,None, None,None 

    # Build subgraph
    subgraph_nodes = set(path)
    if second_hop:
        for node in path:
            for neighbor in G.neighbors(node):
                subgraph_nodes.add(neighbor)
                if G.has_node(neighbor):
                    for second_hop_neighbor in G.neighbors(neighbor):
                        subgraph_nodes.add(second_hop_neighbor)

    subgraph = G.subgraph(subgraph_nodes).copy()
    
    if save_files:
        time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
        nt = Network('500px', '1000px', notebook=True)
        
        # Add nodes and edges from the subgraph to the Pyvis network
        nt.from_nx(subgraph)
        
        fname=f'{data_dir}/shortest_path_2hops_{time_part}_{source}_{target}.html'
        nt.show(fname)
        if verbatim:
            print(f"HTML visualization: {fname}")

        graph_GraphML = f'shortestpath_2hops_{time_part}_{source}_{target}.graphml'
        
        save_graph_without_text(subgraph, data_dir=data_dir, graph_name=graph_GraphML)

        
        if verbatim:
            print(f"GraphML file: {graph_GraphML}")  
    else:
        fname=None
        graph_GraphML=None
    shortest_path_length = len(path) - 1  # As path length is number of edges        

    return path, subgraph, shortest_path_length, fname, graph_GraphML



def find_shortest_path (G,source='graphene', target='complexity', verbatim=True, data_dir='./'):
    
    # Find the shortest path between two nodes
    path = nx.shortest_path(G, source=source, target=target)
    
    shortest_path_length = nx.shortest_path_length(G, source=source, target=target)
    
    path_graph = G.subgraph(path)
    
    nt = Network('500px', '1000px', notebook=True)
    
    # Create a subgraph for the current path
    path_graph = G.subgraph(path)
    
    # Add nodes and edges from the subgraph to the Pyvis network
    nt.from_nx(path_graph)
    
    # Provide a title for the network
    fname=f'{data_dir}/shortest_path_{source}_{target}.html'
    nt.show(fname)
    if verbatim:
        print(f"Visualization: {fname}")
    graph_GraphML = f'{data_dir}/shortestpath_{source}_{target}.graphml'
    nx.write_graphml(path_graph, graph_GraphML)

    return path, path_graph , shortest_path_length, fname, graph_GraphML
def find_shortest_path_with2hops (G, source='graphene', target='complexity',
                                 second_hop=True,#otherwise just neighbors
                                  verbatim=True,data_dir='./', save_files=True,
                                 ):
    # Find the shortest path between two nodes
    path = nx.shortest_path(G, source=source, target=target)
    
    # Initialize a set to keep track of all nodes within 2 hops
    nodes_within_2_hops = set(path)
    
    # Expand the set to include all neighbors within 2 hops of the path nodes
    for node in path:
        for neighbor in G.neighbors(node):
            nodes_within_2_hops.add(neighbor)
            # Include the neighbors of the neighbor (2 hops)

            if second_hop:
                for second_neighbor in G.neighbors(neighbor):
                    nodes_within_2_hops.add(second_neighbor)
    
    # Create a subgraph for the nodes within 2 hops
    path_graph = G.subgraph(nodes_within_2_hops)

    if save_files:
        nt = Network('500px', '1000px', notebook=True)
        
        # Add nodes and edges from the subgraph to the Pyvis network
        nt.from_nx(path_graph)
        
        fname=f'{data_dir}/shortest_path_2hops_{source}_{target}.html'
        nt.show(fname)
        if verbatim:
            print(f"HTML visualization: {fname}")

        graph_GraphML = f'{data_dir}/shortestpath_2hops_{source}_{target}.graphml'
        nx.write_graphml(path_graph, graph_GraphML)
        if verbatim:
            print(f"GraphML file: {graph_GraphML}")  
    else:
        fname=None
        graph_GraphML=None
        
    shortest_path_length = len(path) - 1  # As path length is number of edges
    
    return path, path_graph , shortest_path_length, fname, graph_GraphML

    
def find_N_paths (G, source='graphene', target='complexity', N=5):
    
    sampled_paths = []
    fname_list=[]
    
    # Use a generator to find simple paths and collect up to num_sampled_paths
    paths_generator = nx.all_simple_paths(G, source=source, target=target)
    for _ in range(N):
        try:
            path = next(paths_generator)
            sampled_paths.append(path)
        except StopIteration:
            # This catches the case where there are fewer paths than num_sampled_paths
            break
    
    # Now visualize each sampled path using Pyvis
    for i, path in enumerate(sampled_paths):
        # Create a new Pyvis network for each path
        nt = Network('500px', '1000px', notebook=True)
        
        # Create a subgraph for the current path
        path_graph = G.subgraph(path)
        
        # Add nodes and edges from the subgraph to the Pyvis network
        nt.from_nx(path_graph)
        
        # Provide a title for the network and save to an HTML file
        fname=f'{data_dir}/shortest_path_{source}_{target}_{i}.html'
        
        nt.show(fname)
        print(f"Path {i+1} Visualization: {fname}")
        fname_list.append (fname)

    return sampled_paths, fname_list#, sampled_path_lengths, 
        
from itertools import combinations

def find_all_triplets(G):
    triplets = []
    for nodes in combinations(G.nodes(), 3):
        subgraph = G.subgraph(nodes)
        if nx.is_connected(subgraph) and subgraph.number_of_edges() == 3:
            # Add the triplet to the list as a string
            triplets.append(f"{nodes[0]}-{nodes[1]}-{nodes[2]}")
    return triplets

def print_node_pairs_edge_title(G):
    pairs_and_titles = []
    for node1, node2, data in G.edges(data=True):
        # Assuming 'title' is the edge attribute you want to print
        title = data.get('title', 'No title')  # Default to 'No title' if not present
        pairs_and_titles.append(f"{node1}, {title}, {node2}")
    #print ("Format: node_1, relationship, node_2")
    return pairs_and_titles

def find_path( G, node_embeddings,  tokenizer, model, keyword_1 = "music and sound", keyword_2 = "graphene", 
              verbatim=True, second_hop=False,data_dir='./', similarity_fit_ID_node_1=0, similarity_fit_ID_node_2=0,save_files=True,
              ):
    
    best_node_1, best_similarity_1=find_best_fitting_node_list(keyword_1, node_embeddings, tokenizer, model, max (5, similarity_fit_ID_node_1+1))[similarity_fit_ID_node_1]
    
    if verbatim:
        print(f"{similarity_fit_ID_node_1}nth best fitting node for '{keyword_1}': '{best_node_1}' with similarity: {best_similarity_1}")
    
    
    best_node_2, best_similarity_2 = find_best_fitting_node_list(keyword_2, node_embeddings, tokenizer, model,  max (5, similarity_fit_ID_node_2+1))[similarity_fit_ID_node_2]
    if verbatim:
        print(f"{similarity_fit_ID_node_2}nth best fitting node for '{keyword_2}': '{best_node_2}' with similarity: {best_similarity_2}")
    
    path, path_graph , shortest_path_length, fname, graph_GraphML= find_shortest_path_with2hops (G,
                                source=best_node_1, target=best_node_2, second_hop=second_hop, verbatim=verbatim, data_dir=data_dir,save_files=save_files,
                                                                                                 )
    
    
    return (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML

import community as community_louvain
import math

def describe_communities(G, N=10):
    """
    Detect and describe the top N communities in graph G based on key nodes.
    
    Args:
    - G (networkx.Graph): The graph to analyze.
    - N (int): The number of top communities to describe.
    """
    # Detect communities using the Louvain method
    partition = community_louvain.best_partition(G)

    # Invert the partition to get nodes per community
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # Sort communities by size
    sorted_communities = sorted(communities.items(), key=lambda item: len(item[1]), reverse=True)[:N]

    # Describe each of the top N communities
    for i, (comm_id, nodes) in enumerate(sorted_communities, start=1):
        subgraph = G.subgraph(nodes)
        degrees = subgraph.degree()
        sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
        key_nodes = sorted_nodes[:5]  # Adjust the number of key nodes as needed

        # Print community description
        print(f"Community {i} (ID {comm_id}) with {len(nodes)} nodes, key nodes (Node ID: Degree):")
        for node_id, degree in key_nodes:
            print(f"  Node {node_id}: Degree {degree}")
        # Add more analysis as needed, e.g., other centrality measures, node attributes

def describe_communities_with_plots(G, N=10, N_nodes=5, data_dir='./'):
    """
    Detect and describe the top N communities in graph G based on key nodes, with integrated plots.
    
    Args:
    - G (networkx.Graph): The graph to analyze.
    - N (int): The number of top communities to describe and plot.
    - data_dir (str): Directory to save the plots.
    """
    # Detect communities using the Louvain method
    partition = community_louvain.best_partition(G)

    # Invert the partition to get nodes per community
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # Sort communities by size and get all sizes
    all_communities_sorted = sorted(communities.values(), key=len, reverse=True)
    all_sizes = [len(c) for c in all_communities_sorted]
    
    # Plot sizes of all communities
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(all_sizes)), all_sizes, color='skyblue')
    plt.title('Size of All Communities')
    plt.xlabel('Community Index')
    plt.ylabel('Size (Number of Nodes)')
    plt.savefig(f'{data_dir}/size_of_communities.svg')
    plt.show()

    # Determine subplot grid size
    rows = math.ceil(N / 2)
    cols = 2 if N > 1 else 1  # Use 2 columns if N > 1, else just 1

    # Create integrated plot with subplots for top N communities
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols,10 * rows), squeeze=False)
    for i, nodes in enumerate(all_communities_sorted[:N], start=0):
        subgraph = G.subgraph(nodes)
        degrees = dict(subgraph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        key_nodes, key_degrees = zip(*sorted_nodes[:N_nodes])  # Adjust number as needed

        # Select the appropriate subplot
        ax = axes[i // cols, i % cols]
        bars = ax.bar(range(len(key_nodes)), key_degrees, tick_label=key_nodes, color='lightgreen')
        ax.set_title(f'Community {i+1} (Top Nodes by Degree)', fontsize=18)
        ax.set_xlabel('Node label', fontsize=18)
        ax.set_ylabel('Degree', fontsize=18)
        ax.tick_params(axis='x', labelsize=18, rotation=45)
        ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.savefig(f'{data_dir}/top_nodes_by_degree_combined.svg')
    plt.show()

def describe_communities_with_plots_complex (G, N=10, N_nodes=5, data_dir='./'):
    """
    Detect and describe the top N communities in graph G based on key nodes, with integrated plots.
    Adds separate plots for average node degree, average clustering coefficient, and betweenness centrality over all communities.
    
    Args:
    - G (networkx.Graph): The graph to analyze.
    - N (int): The number of top communities to describe and plot.
    - N_nodes (int): The number of top nodes to highlight per community.
    - data_dir (str): Directory to save the plots.
    """
    # Detect communities using the Louvain method
    partition = community_louvain.best_partition(G)

    # Invert the partition to get nodes per community
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # Sort communities by size and get all sizes
    all_communities_sorted = sorted(communities.values(), key=len, reverse=True)
    all_sizes = [len(c) for c in all_communities_sorted]

    # Arrays to hold statistics for all communities
    avg_degrees = []
    avg_clusterings = []
    top_betweenness_values = []

    # Calculate statistics for each community
    for nodes in all_communities_sorted:
        subgraph = G.subgraph(nodes)
        degrees = dict(subgraph.degree())
        avg_degree = np.mean(list(degrees.values()))
        avg_clustering = nx.average_clustering(subgraph)
        betweenness = nx.betweenness_centrality(subgraph)
        top_betweenness = sorted(betweenness.values(), reverse=True)[:N_nodes]

        avg_degrees.append(avg_degree)
        avg_clusterings.append(avg_clustering)
        top_betweenness_values.append(np.mean(top_betweenness) if top_betweenness else 0)

    # Create integrated plot with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))  # Adjust for a 2x2 subplot layout

    # Plot size of all communities
    axs[0, 0].bar(range(len(all_sizes)), all_sizes, color='skyblue')
    axs[0, 0].set_title('Size of All Communities')
    axs[0, 0].set_xlabel('Community Index')
    axs[0, 0].set_ylabel('Size (Number of Nodes)')

    # Plot average node degree for each community
    axs[0, 1].bar(range(len(avg_degrees)), avg_degrees, color='lightgreen')
    axs[0, 1].set_title('Average Node Degree for Each Community')
    axs[0, 1].set_xlabel('Community Index')
    axs[0, 1].set_ylabel('Average Degree')

    # Plot average clustering coefficient for each community
    axs[1, 0].bar(range(len(avg_clusterings)), avg_clusterings, color='lightblue')
    axs[1, 0].set_title('Average Clustering Coefficient for Each Community')
    axs[1, 0].set_xlabel('Community Index')
    axs[1, 0].set_ylabel('Average Clustering Coefficient')

    # Plot average betweenness centrality for top nodes in each community
    axs[1, 1].bar(range(len(top_betweenness_values)), top_betweenness_values, color='salmon')
    axs[1, 1].set_title('Average Betweenness Centrality for Top Nodes in Each Community')
    axs[1, 1].set_xlabel('Community Index')
    axs[1, 1].set_ylabel('Average Betweenness Centrality')

    plt.tight_layout()
    plt.savefig(f'{data_dir}/community_statistics_overview.svg')
    plt.show()

    # Determine subplot grid size
    rows = math.ceil(N / 2)
    cols = 2 if N > 1 else 1  # Use 2 columns if N > 1, else just 1

    # Create integrated plot with subplots for top N communities
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols,12 * rows), squeeze=False)
    for i, nodes in enumerate(all_communities_sorted[:N], start=0):
        subgraph = G.subgraph(nodes)
        degrees = dict(subgraph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        key_nodes, key_degrees = zip(*sorted_nodes[:N_nodes])  # Adjust number as needed

        # Select the appropriate subplot
        ax = axes[i // cols, i % cols]
        bars = ax.bar(range(len(key_nodes)), key_degrees, tick_label=key_nodes, color='lightgreen')
        ax.set_title(f'Community {i+1} (Top Nodes by Degree)', fontsize=18)
        ax.set_xlabel('Node label', fontsize=18)
        ax.set_ylabel('Degree', fontsize=18)
        ax.tick_params(axis='x', labelsize=18, rotation=45)
        ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.savefig(f'{data_dir}/top_nodes_by_degree_combined.svg')
    plt.show()

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
import networkx as nx
import powerlaw
import matplotlib.pyplot as plt

def is_scale_free_simple(G, plot_distribution=True, data_dir='./'):
    """
    Determines if the network G is scale-free using the powerlaw package.
    
    Args:
    - G (networkx.Graph): The graph to analyze.
    - plot_distribution (bool): If True, plots the degree distribution with the power-law fit.
    
    Returns:
    - bool indicating if the network is scale-free.
    - Fit object from the powerlaw package.
    """
    degrees = sorted([d for n, d in G.degree() if d > 0], reverse=True)
    fit = powerlaw.Fit(degrees, discrete=True)
    alpha = fit.power_law.alpha
    sigma = fit.power_law.sigma

    if plot_distribution:
        plt.figure(figsize=(10, 6))
        fig = fit.plot_pdf(color='b', linewidth=2)
        fit.power_law.plot_pdf(color='r', linestyle='--', linewidth=2)
        plt.xlabel('Degree')
        plt.ylabel('Probability')
        plt.title('Degree Distribution with Power-law Fit')
        plt.savefig(f'{data_dir}/degree_dist_with_powerlaw_fit.svg')
        plt.show()

    # Use the distribution comparison method provided by the powerlaw package
    R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    # If R > 0 and p < 0.05, the power-law model is a better fit to the data than the exponential model
    is_scale_free = R > 0 and p < 0.05

    print(f"Power-law exponent (alpha): {alpha}")
    print(f"Standard error of alpha: {sigma}")
    print(f"Loglikelihood ratio (R) comparing power-law to exponential: {R}")
    print(f"p-value for the comparison: {p}")

    return is_scale_free, fit
     
def is_scale_free(G, plot_distribution=True, data_dir='./', manual_xmin=None):
    """
    Determines if the network G is scale-free using the powerlaw package.
    
    Args:
    - G (networkx.Graph): The graph to analyze.
    - plot_distribution (bool): If True, plots the degree distribution with the power-law fit.
    - data_dir (str): Directory to save the plots.
    - manual_xmin (int): Manually set the xmin value for power-law fitting.
    
    Returns:
    - bool indicating if the network is scale-free.
    - Fit object from the powerlaw package.
    """
    # Get degrees and sort
    degrees = sorted([d for n, d in G.degree() if d > 0], reverse=True)

    
    # Fit the power-law model
    if manual_xmin:
        fit = powerlaw.Fit(degrees, discrete=True, xmin=manual_xmin)
    else:
        fit = powerlaw.Fit(degrees, discrete=True)
    
    if plot_distribution:
        plt.figure(figsize=(10, 6))
        fit.plot_pdf(color='b', linewidth=2)
        fit.power_law.plot_pdf(color='r', linestyle='--', linewidth=2)
        plt.xlabel('Degree')
        plt.ylabel('Probability')
        plt.title('Degree Distribution with Power-law Fit')
        plt.savefig(f'{data_dir}/degree_dist_with_powerlaw_fit.svg')
        plt.show()

    # Compare the power-law fit to alternative distributions
    R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    
    # Print the fit parameters and comparison results
    print(f"Power-law exponent (alpha): {fit.power_law.alpha}")
    print(f"Standard error of alpha: {fit.power_law.sigma}")
    print(f"Loglikelihood ratio (R) comparing power-law to exponential: {R}")
    print(f"p-value for the comparison: {p}")

    # Determine if the network is scale-free
    is_scale_free = R > 0 and p < 0.05

    return is_scale_free, fit

def print_path_with_edges_as_list_multigraph (G, path, keywords_separator=' --> '):
    path_elements = []

    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]

        # Retrieve edge data between node1 and node2
        edge_data = G.get_edge_data(node1, node2)

        # If there are multiple edges, choose the 'title' from the first one (or customize as needed)
        if edge_data:
            titles = [data.get('title', 'No title') for _, data in edge_data.items()]
            edge_title = " or ".join(titles)  # This handles multiple edges by joining their titles with " or "
        else:
            edge_title = 'No title'
        # Construct the path elements, inserting the edge title between node pairs
        if i == 0:
            path_elements.append(node1)  # Add the first node
        path_elements.append(edge_title)  # Add the edge title
        path_elements.append(node2)  # Add the second node

    # Convert the list of path elements into a string with the specified separator
    as_string = keywords_separator.join(path_elements)

    return path_elements, as_string

def print_path_with_edges_as_list(G, path, keywords_separator=' --> '):
    path_elements = []

    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]

        # Retrieve edge data between node1 and node2
        edge_data = G.get_edge_data(node1, node2)

        # Access the 'title' directly from the edge_data
        if edge_data:
            edge_title = edge_data.get('title', 'No title')
        else:
            edge_title = 'No title'

        # Construct the path elements, inserting the edge title between node pairs
        if i == 0:
            path_elements.append(node1)  # Add the first node
        path_elements.append(edge_title)  # Add the edge title
        path_elements.append(node2)  # Add the second node

    # Convert the list of path elements into a string with the specified separator
    as_string = keywords_separator.join(path_elements)

    return path_elements, as_string
    
def find_path_and_reason (G, node_embeddings,  tokenizer, model, generate, 
                          keyword_1 = "music and sound",
                         # local_llm=None,
                          keyword_2 = "apples",include_keywords_as_nodes=True,
                          inst_prepend='',
                          graph_analysis_type='path and relations',# 'keywords', #0=path defined by nodes, 1=triplets, 2=pairs
                          instruction = 'Now, reason over them and propose a research hypothesis.',
                          verbatim=False,
                          N_limit=None,temperature=0.3,
                          keywords_separator=' --> ',system_prompt='You are a scientist who uses logic and reasoning.',
                          max_tokens=4096,prepend='You are given a set of information from a graph that describes the relationship between materials, structure, properties, and properties. You analyze these logically through reasoning.\n\n',
                          similarity_fit_ID_node_1=0, similarity_fit_ID_node_2=0, #whoch path to include 0=only best, 1 onlysecond best, etc.
                          save_files=True,data_dir='./',visualize_paths_as_graph=True, display_graph=True,words_per_line=2,
                         ):
    make_dir_if_needed(data_dir)
    task=prepend+''

    join_strings = lambda strings: '\n'.join(strings)
    join_strings_newline = lambda strings: '\n'.join(strings)

    
     
    (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML=find_path( G,node_embeddings, tokenizer, model, keyword_1 = keyword_1,  keyword_2 = keyword_2,verbatim=verbatim,
                                                              similarity_fit_ID_node_1=similarity_fit_ID_node_1,similarity_fit_ID_node_2=similarity_fit_ID_node_2,  data_dir=data_dir,
                                                                                                                                             save_files=save_files,)
    if visualize_paths_as_graph:
        path_list_for_vis, _=path_list=print_path_with_edges_as_list(G, path, keywords_separator=keywords_separator)                                                                                                                                        
    if include_keywords_as_nodes:
        if keyword_1!=best_node_1:
            path.insert (0, keyword_1)
            
        if keyword_2!=best_node_2:
            path.append (keyword_2) 

    
    
    if contains_phrase( graph_analysis_type, 'keywords') :
        if N_limit != None:
            path=path[:N_limit]
        join_strings = lambda strings: keywords_separator.join(strings)
        
        task=task+f"{inst_prepend}Consider these keywords:\n\n{join_strings( path)}\n\nThese keywords form a path in a knowledge graph between {keyword_1} and {keyword_2}.\n\n"

    if contains_phrase( graph_analysis_type, 'path and relations') :
        if N_limit != None:
            path=path[:N_limit]
        _, path_list_string=path_list=print_path_with_edges_as_list(G, path, keywords_separator=keywords_separator)

        if visualize_paths_as_graph:
            
            if include_keywords_as_nodes:
                if keyword_1!=best_node_1:
                    path_list_for_vis.insert (0, keyword_1)
                    path_list_for_vis.insert (1, '')
                    
                    
                if keyword_2!=best_node_2:
                    path_list_for_vis.append ('')
                    path_list_for_vis.append (keyword_2)
                
            if verbatim:
                print ("Raw path for graph visualization: ", path_list_for_vis, 'original: ', path)
            
            G_vis=visualize_paths_pretty([path_list_for_vis], filename=f'{best_node_1}_{best_node_2}.svg', display_graph=display_graph,data_dir=data_dir, scale=1.25, node_size=4000,words_per_line=words_per_line)
            nx.write_graphml(G_vis, f'{data_dir}/{best_node_1}_{best_node_2}.graphml')
            make_HTML (G_vis,data_dir=data_dir, graph_root=f'{best_node_1}_{best_node_2}')
        
        task=task+f"{inst_prepend}Consider these nodes and their relations, forming a path:\n\n{path_list_string}\n\nThese keywords form a path in a knowledge graph between {keyword_1} and {keyword_2}, along with their edges that describe the relationship between nodes.\n\n"

    
    if contains_phrase( graph_analysis_type, 'triplets'):
        triplets=find_all_triplets(path_graph)
        if N_limit != None:
            triplets=triplets[:N_limit]
        
        task=task+f"{inst_prepend}Consider these graph triplets extracted from a knowledge graph:\n\n{join_strings( triplets)}\n\nThese are triplets from a knowledge graph between {keyword_1} and {keyword_2}.\n\n"
    
    if contains_phrase( graph_analysis_type, 'nodes and relations'):
        node_list=print_node_pairs_edge_title(path_graph)
        if N_limit != None:
            node_list=node_list[:N_limit]
        
        task=task+f"{inst_prepend}Consider this list of nodes and relations in a knowledge graph:\n\nFormat: node_1, relationship, node_2\n\nThe data is:\n\n{join_strings_newline( node_list)}\n\nThese are from a knowledge graph between {keyword_1} and {keyword_2}.\n\n"

    task=task+f"{inst_prepend}{instruction}"
    
    print ( task)
    
    response=generate(system_prompt=system_prompt, #local_llm=local_llm,
         prompt=task, max_tokens=max_tokens, temperature=temperature)
    
    display(Markdown("**Response:** "+response ))

    return response , (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML


def find_path_with_relations_and_reason_combined(G, node_embeddings, tokenizer, model, generate,
                          keyword_1="music and sound",
                          
                          keyword_2="apples", include_keywords_as_nodes=True,
                          inst_prepend='',
                          instruction='Now, reason over them and propose a research hypothesis.',
                          verbatim=False,
                          N_limit=None, temperature=0.3,
                          keywords_separator=' --> ',
                          system_prompt='You are a scientist who uses logic and reasoning.',
                          max_tokens=4096,
                          prepend='You are given a set of information from a graph that describes the relationship between materials, structure, properties, and properties. You analyze these logically through reasoning.\n\n',
                          num_paths=2,  # Used for both single path per ID and all combinations
                          include_all_possible=False,  # New option to consider all combinations,
                          data_dir='./',save_files=False, #whether or not to make HTML of each graph
                           visualize_paths_as_graph=False, display_graph=True,words_per_line=2,
                         ):

    make_dir_if_needed(data_dir)
    task = prepend + ''
    first=True
     
    complete_path_list=[]#for integrated path

    paths_details = []  # List to store details of each path

    if include_all_possible:
        # Iterate through all combinations of similarity_fit_ID_node values
        for start_id in range(num_paths):
            for end_id in range(num_paths):
                # Process each combination
                 
                paths_details.extend(process_path_combination(G, node_embeddings, tokenizer, model, keyword_1, keyword_2, verbatim, N_limit, keywords_separator, start_id, end_id, include_keywords_as_nodes,data_dir,save_files,visualize_paths_as_graph,display_graph=display_graph,words_per_line=words_per_line))
    else:
        # Process paths based on num_paths without considering all combinations
        for path_id in range(num_paths):

             
            paths_details.extend(process_path_combination(G, node_embeddings, tokenizer, model, keyword_1, keyword_2, verbatim, N_limit, keywords_separator, path_id, path_id, include_keywords_as_nodes,data_dir,save_files,visualize_paths_as_graph,display_graph=display_graph,words_per_line=words_per_line))
             

    # Generate task and response for each path or combination
    for i, (start_id, end_id, path_list, path_list_vis, path_list_string) in enumerate(paths_details):
        
        complete_path_list.append (path_list_vis)
        if i == 0:  # First combination
            task += f"{inst_prepend}Primary combination (path from {start_id} to {end_id}):\n\n{path_list_string}\n\nThis represents the main combination of nodes in the knowledge graph between {keyword_1} and {keyword_2}.\n\n"
        else:  # Subsequent combinations
            if first:
                task+="The following represent another possible combination of paths, providing different insights or complementing the primary path.\n\n"
                first=False
            task += f"{inst_prepend}Alternative combination (path from {start_id} to {end_id}):\n\n{path_list_string}\n\n"

    task += f"{inst_prepend}{instruction}\n\n"
    if visualize_paths_as_graph:
        G_vis=visualize_paths_and_save_with_labels(complete_path_list, filename=f'joined_{keyword_1[:20]}_{keyword_2[:20]}.svg', display_graph=display_graph,data_dir=data_dir, scale=1.25, node_size=4000,words_per_line=words_per_line)
        nx.write_graphml(G_vis, f'{data_dir}/joined_{keyword_1[:20]}_{keyword_2[:20]}.graphml')
        make_HTML (G_vis,data_dir=data_dir, graph_root=f'joined_{keyword_1[:20]}_{keyword_2[:20]}')
        
        G_vis=visualize_paths_unique(complete_path_list, filename=f'joined_unique_{keyword_1[:20]}_{keyword_2[:20]}.svg', display_graph=display_graph,data_dir=data_dir, scale=1.25, node_size=4000,words_per_line=words_per_line)
        nx.write_graphml(G_vis, f'{data_dir}/joined_unique_{keyword_1[:20]}_{keyword_2[:20]}.graphml')
    
        make_HTML (G_vis,data_dir=data_dir, graph_root=f'joined_unique_{keyword_1[:20]}_{keyword_2[:20]}')

    print ( task)
    
    # Generate response based on the task
    response = generate(system_prompt=system_prompt,# local_llm=local_llm,
                        prompt=task, max_tokens=max_tokens, temperature=temperature)

    display(Markdown("**Response:** " + response))

    

    return response

def process_path_combination(G, node_embeddings, tokenizer, model, keyword_1, keyword_2, verbatim, N_limit, keywords_separator, start_id, end_id, include_keywords_as_nodes,data_dir,save_files,
                            visualize_paths_as_graph=False,display_graph=False,words_per_line=2):
    # This helper function encapsulates the repeated logic for finding and processing a path
    paths_details = []
    (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML = find_path(
        G, node_embeddings, tokenizer, model, keyword_1=keyword_1,
        keyword_2=keyword_2, verbatim=verbatim,
        similarity_fit_ID_node_1=start_id,
        similarity_fit_ID_node_2=end_id,data_dir=data_dir,save_files=save_files,
    )

    if visualize_paths_as_graph:
        path_new, _ = print_path_with_edges_as_list(G, path, keywords_separator=keywords_separator)
         
        if include_keywords_as_nodes:
            if keyword_1!=best_node_1:
                path_new.insert (0, keyword_1)
                path_new.insert (1, '')
                
                
            if keyword_2!=best_node_2:
                path_new.append ('')
                path_new.append (keyword_2)
        _=visualize_paths_pretty([path_new], filename=f'{best_node_1}_{best_node_2}.svg', display_graph=display_graph,data_dir=data_dir, scale=1.25, node_size=4000,words_per_line=words_per_line)
    
    # Include keywords as nodes if not already part of the path
    if include_keywords_as_nodes:
        if keyword_1 != best_node_1:
            path.insert(0, keyword_1)
            #path.insert (1, '')
        if keyword_2 != best_node_2:
           # path.append ('')
            path.append(keyword_2)

    # Limit the number of nodes in the path if N_limit is specified
    if N_limit is not None:
        path = path[:N_limit]

    # Generate path list and string representation
    path_list, path_list_string = print_path_with_edges_as_list(G, path, keywords_separator=keywords_separator)
    path_list_vis, _ = print_path_with_edges_as_list(G, path_new, keywords_separator=keywords_separator)

    # Store path details
    paths_details.append((start_id, end_id, path_list, path_list_vis, path_list_string))
    return paths_details

import networkx as nx
import matplotlib.pyplot as plt

def split_label(label, words_per_line=3):
    """Splits a label into multiple lines, with a specified number of words per line."""
    words = label.split()
    # Split words into chunks of `words_per_line`
    lines = [' '.join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]
    return '\n'.join(lines)
    
def visualize_paths_and_save(paths, filename='graph.svg', display_graph=False, data_dir='./',scale=1.25, node_size=4000,):
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add edges for each path in the list of paths
    for path in paths:
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1])

    # Position nodes using the spring layout
    pos = nx.spring_layout(G, seed=42, k=0.75, iterations=100)

    # Draw the graph
    plt.figure(figsize=(12, 8))  # Set the size of the figure
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, 
            edge_color='k', linewidths=1*scale, font_size=10*scale, 
            arrows=True, arrowsize=20*scale)
    
    # Save the graph as an SVG file
    plt.savefig(data_dir+filename, format='svg')
    
    # Display the graph in the Jupyter Notebook if requested
    if display_graph:
        plt.show()
    else:
        plt.close()  # Close the plot to prevent it from displaying unnecessarily

def visualize_paths_unique(paths, filename='graph_distinguish.svg', display_graph=False, data_dir='./', scale=1.25, node_size=4000, words_per_line=2):
    G = nx.DiGraph()
    node_labels = {}  # For storing original labels for unique nodes

    # Collect all start and end nodes across paths for special handling
    all_starts_ends = {p[0]: 0 for p in paths}  # Use dict to preserve order
    all_starts_ends.update({p[-1]: 0 for p in paths})

    # Function to split label into multiple lines
    def split_label(label, words_per_line=2):
        words = label.split()
        return '\n'.join([' '.join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)])

    # Assigning unique identifiers and tracking start/end nodes
    for path_index, path in enumerate(paths):
        for i in range(0, len(path) - 2, 2):
            relationship = split_label(path[i + 1], words_per_line=words_per_line).lower()
            # Determine if the node is a unique start/end node
            start_node_key = f"{path[i]}_{path_index}" if path[i] not in all_starts_ends else path[i]
            end_node_key = f"{path[i+2]}_{path_index}" if path[i+2] not in all_starts_ends else path[i+2]
            
            G.add_edge(start_node_key, end_node_key, label=relationship)
            node_labels[start_node_key] = path[i]
            node_labels[end_node_key] = path[i+2]

    # Position nodes using the Kamada-Kawai layout for better appearance
    pos = nx.spring_layout(G, seed=42, k=3./scale, iterations=150)

    plt.figure(figsize=(15 * scale, 10 * scale))

    # Determine node colors: start/end nodes one color, others a different color
    node_colors = ['lightgreen' if node in all_starts_ends else 'skyblue' for node in G.nodes()]
    
    # Draw nodes, edges, and labels
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_size, edge_color='k', linewidths=1*scale, font_size=10*scale, arrows=True, arrowsize=20*scale)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10*scale, font_color='darkblue')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10*scale, font_color='red')
    
    plt.axis('off')
    plt.savefig(data_dir + filename, format='svg')
    if display_graph:
        plt.show()

    return G
def visualize_paths_and_save_with_labels(paths, filename='graph.svg', display_graph=False, data_dir='./',scale=1.25, node_size=4000,words_per_line=2):
    '''
    paths must be alternativing node - relationship - node - relationship - node
    '''
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges with labels for each path in the list of paths
    for path in paths:
        for i in range(0, len(path) - 2, 2):  # Step by 2 to skip to the next node
            node_start = path[i]
            relationship = path[i + 1]
            node_end = path[i + 2]
            relationship = split_label(relationship,words_per_line=words_per_line)
            G.add_edge(node_start.lower(), node_end.lower(), label=relationship.lower())

    # Position nodes using the spring layout
    pos = nx.spring_layout(G, seed=42, k=0.75, iterations=100)

    # Draw the graph
    plt.figure(figsize=(15, 10))  # Set the size of the figure
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, 
            edge_color='k', linewidths=1*scale, font_size=10*scale, 
            arrows=True, arrowsize=20*scale )

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=10*scale)
    
    # Save the graph as an SVG file
    plt.savefig(data_dir+filename, format='svg')
    
    # Display the graph in the Jupyter Notebook if requested
    if display_graph:
        plt.show()
    else:
        plt.close()  # Close the plot to prevent it from displaying unnecessarily

    return G

def visualize_paths_pretty(paths, filename='graph_pretty.svg', display_graph=False, data_dir='./', scale=1.25, node_size=4000, words_per_line=2):
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Function to split labels over multiple lines
    def split_label(label, words_per_line=2):
        words = label.split()
        return '\n'.join([' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)])
    
    # Iterate through each path and add nodes and edges accordingly
    pos = {}
    node_positions = []  # Keep track of node positions for ordering

    for path in paths:
        for i in range(0, len(path) - 2, 2):  # Step by 2 to process node, edge, node triples
            node_start = path[i].lower()
            relationship = split_label(path[i + 1], words_per_line=words_per_line).lower()
            node_end = path[i + 2].lower()

            # Add edge with label
            G.add_edge(node_start, node_end, label=relationship)

            # Track node positions if not already tracked
            if node_start not in node_positions:
                node_positions.append(node_start)
            if i + 2 == len(path) - 1 and node_end not in node_positions:  # Ensure the last node is added
                node_positions.append(node_end)

    # Assign positions based on the order of appearance
    pos = {node: (index, 0) for index, node in enumerate(node_positions)}

    # Draw the graph
    plt.figure(figsize=(15, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=node_size, edge_color='gray', linewidths=1*scale, font_size=10*scale, arrows=True, arrowsize=20*scale, alpha=0.8)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10*scale)

    # Save and/or display the graph
    plt.savefig(data_dir + filename, format='svg')
    if display_graph:
        plt.show()
    else:
        plt.close()

    return G

def visualize_paths_pretty_spiral(paths, filename='graph_pretty.svg', display_graph=False,data_dir='./', scale=1.25, node_size=4000,words_per_line=2):
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges with labels for each path in the list of paths
    for path in paths:
        for i in range(0, len(path) - 2, 2):  # Step by 2 to skip to the next node
            node_start = path[i]
            relationship = path[i + 1]
            node_end = path[i + 2]
            relationship = split_label(relationship,words_per_line=words_per_line)
            G.add_edge(node_start.lower(), node_end.lower(), label=relationship.lower())

    # Use shell layout for a more structured arrangement
    # Calculate number of shells based on the longest path
    num_shells = max((len(path) + 1) // 2 for path in paths)
    shells = []
    for i in range(num_shells):
        shells.append([node  for path in paths for node in path[i*2::2] if i*2 < len(path)])
    pos = nx.shell_layout(G, shells)

    # Draw the graph with the shell layout
    plt.figure(figsize=(15, 10))  # Set the size of the figure
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=node_size, 
            edge_color='gray', linewidths=1*scale, font_size=10*scale, 
            arrows=True, arrowsize=20*scale, alpha=0.8 )

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10*scale)
    
    # Save the graph as an SVG file
    plt.savefig(data_dir+filename, format='svg')
    
    # Display the graph in the Jupyter Notebook if requested
    if display_graph:
        plt.show()
    else:
        plt.close()  # Close the plot to prevent it from displaying unnecessarily

    return G

import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
import community as community_louvain  # This is the python-louvain package

def analyze_and_visualize_community_structure(G, data_dir='./', root='graph', algorithm='greedy_modularity'):
    os.makedirs(data_dir, exist_ok=True)
    
    if algorithm == 'greedy_modularity':
        communities = list(greedy_modularity_communities(G))
    elif algorithm == 'louvain':
        partition = community_louvain.best_partition(G)
        # Convert partition format to a list of sets
        temp_communities = {}
        for node, comm_id in partition.items():
            if comm_id not in temp_communities:
                temp_communities[comm_id] = set()
            temp_communities[comm_id].add(node)
        communities = list(temp_communities.values())
    else:
        raise ValueError("Unsupported algorithm. Use 'greedy_modularity' or 'louvain'")
    
    # Sort communities by size
    communities.sort(key=len, reverse=True)
    community_sizes = [len(community) for community in communities]
    modularity_score = modularity(G, communities)
    
    intra_community_edges = 0
    inter_community_edges = 0
    for u, v in G.edges():
        if any(u in community and v in community for community in communities):
            intra_community_edges += 1
        elif any(u in community or v in community for community in communities):
            inter_community_edges += 1
    
    avg_intra_community_edges = intra_community_edges / len(communities) if communities else 0
    avg_inter_community_edges = inter_community_edges / len(communities) if communities else 0

    # Prepare the figure for a 2x2 subplot layout
    fig, axs = plt.subplots(4, 1, figsize=(12, 24))  # Adjusted for a 4x1 layout
    axs = axs.flatten()  # Flatten the array to access subplots linearly
    
    # Plot 1: Community Size Distribution
    axs[0].bar(range(len(community_sizes)), community_sizes, color='skyblue')
    axs[0].set_title('Community Size Distribution', fontsize=12)
    axs[0].set_xlabel('Community Index', fontsize=12)
    axs[0].set_ylabel('Size', fontsize=12)
    
    # Plot 2: Modularity Score
    axs[1].bar(['Modularity'], [modularity_score], color='lightgreen')
    axs[1].set_title('Modularity Score', fontsize=14)
    
    # Plot 3: Intra- and Inter-Community Connectivity
    axs[2].bar(['Avg Intra-Community Edges', 'Avg Inter-Community Edges'], 
               [avg_intra_community_edges, avg_inter_community_edges], color=['skyblue', 'lightgreen'])
    axs[2].set_title('Community Connectivity', fontsize=12)
    axs[2].set_ylabel('Average Number of Edges', fontsize=12)
    
    # Plot 4: Degree Distribution on a Log-Log Scale
    degrees = [G.degree(n) for n in G.nodes()]
    degree_count = Counter(degrees)
    deg, cnt = zip(*degree_count.items())
    total = sum(cnt)
    prob = [c / total for c in cnt]
    axs[3].loglog(deg, prob, marker='o', linestyle='None', color='red')
    axs[3].set_title('Degree Distribution', fontsize=12)
    axs[3].set_xlabel('Degree', fontsize=12)
    axs[3].set_ylabel('Probability', fontsize=12)

    plt.tight_layout()

    # Save the figure
    fig.savefig(os.path.join(data_dir, f"{root}_community_analysis_{algorithm}.svg"), format="svg")

    plt.show()

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import networkx as nx
from tqdm import tqdm
from networkx.algorithms.community import greedy_modularity_communities, modularity
from networkx.algorithms.isomorphism import GraphMatcher
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.cm as cm
import numpy as np
from networkx.algorithms.components import connected_components

def visualize_community_structure_nolabels(G, title="Graph", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    communities = greedy_modularity_communities(G)
    community_map = {node: cid for cid, community in enumerate(communities) for node in community}
    colors = cm.rainbow(np.linspace(0, 1, len(communities)))
    
    pos = graphviz_layout(G, prog='neato')  # Using 'neato' for improved layout
    for community, color in zip(communities, colors):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=[color], ax=ax, node_size=50)  # Adjust node_size if necessary
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    # Omitting labels for clarity or using a custom labeling strategy
    
    ax.set_title(title)
    plt.axis('off')
def get_giant_component(G):
    # This function returns the giant component of a graph G
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(Gcc[0])
def visualize_community_structure(G, title="Graph", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    communities = greedy_modularity_communities(G)
    community_map = {node: cid for cid, community in enumerate(communities) for node in community}
    colors = cm.rainbow(np.linspace(0, 1, len(communities)))
    
    pos = graphviz_layout(G, prog='neato')  # Using 'neato' for improved layout
    
    for community, color in zip(communities, colors):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=[color], ax=ax, node_size=50)
    
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

    # Custom strategy for label positioning to reduce overlap
    label_pos = {k: (v[0], v[1] + 10) for k, v in pos.items()}  # Adjust y offset as needed

    # Draw labels with a smaller font size
    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=6)  # Reduced font size to 6
    
    ax.set_title(title)
    plt.axis('off')

def split_label_forgraph(label, max_words=2):
    """Split a label into multiple lines after 'max_words' words."""
    words = label.split()
    if len(words) <= max_words:
        return label
    else:
        # Split the label into lines with 'max_words' words each
        split_labels = [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
        return '\n'.join(split_labels)

def find_isomorphism_and_map_edges_justnodebased(G1, G2):
    # Create a GraphMatcher object for G1 and G2
    matcher = GraphMatcher(G1, G2)
    
    # Attempt to find an isomorphism between G1 and G2
    if matcher.is_isomorphic():
        # Get the mapping of nodes from G1 to G2
        node_mapping = matcher.mapping
        print("Node Mapping from G1 to G2:")
        print(node_mapping)
        
        # Use the node mapping to directly map edges from G1 to G2
        edge_mapping = {}
        for (u, v) in G1.edges():
            # Get the corresponding nodes in G2 using the node mapping
            u_mapped, v_mapped = node_mapping[u], node_mapping[v]
            # Map the edge in G1 to its corresponding edge in G2 based on the node mapping
            edge_mapping[(u, v)] = (u_mapped, v_mapped)
        
        print("Edge Mapping from G1 to G2:")
        for edge_g1, edge_g2 in edge_mapping.items():
            print(f"{edge_g1} in G1 maps to {edge_g2} in G2")
        return node_mapping, edge_mapping
    else:
        print("The graphs are not isomorphic.")
        return None, None
        
def find_isomorphism_and_map_edges(G1, G2, edge_label='title'):
    # Create a GraphMatcher object for G1 and G2
    matcher = GraphMatcher(G1, G2)
    
    # Attempt to find an isomorphism between G1 and G2
    if matcher.is_isomorphic():
        # Get the mapping of nodes from G1 to G2
        node_mapping = matcher.mapping
        print("Node Mapping from G1 to G2:")
        print(node_mapping)
        
        # Use the node mapping to directly map edges from G1 to G2, including their labels
        edge_mapping = {}
        for (u, v) in G1.edges():
            # Get the corresponding nodes in G2 using the node mapping
            u_mapped, v_mapped = node_mapping[u], node_mapping[v]
            # Assuming the edge exists in G2, retrieve its label
            edge_label_g2 = G2[u_mapped][v_mapped].get(edge_label, "Label not found") if G2.has_edge(u_mapped, v_mapped) else "Label not found"
            # Map the edge in G1 to its corresponding edge in G2 based on the node mapping and include the label
            edge_mapping[(u, v)] = ((u_mapped, v_mapped), G1[u][v].get(edge_label, "Label not found"), edge_label_g2)
        
        print("Edge Mapping from G1 to G2 with Labels:")
        for edge_g1, (edge_g2, label_g1, label_g2) in edge_mapping.items():
            print(f"{edge_g1} ('{label_g1}') in G1 maps to {edge_g2} ('{label_g2}') in G2")
        return node_mapping, edge_mapping
    else:
        print("The graphs are not isomorphic.")
        return None, None

def visualize_community_structure_in_giant_component(G1, G2, title1='Subgraph 1 Giant Component', title2='Subgraph 2 Giant Component', filename='plot.svg',
                                                    data_dir='./', root='graph'):
    plt.figure(figsize=(24, 12))  # Adjust the figure size as needed
    fig, axs = plt.subplots(1, 2, figsize=(24, 12))

    node_mapping, edge_mapping= find_isomorphism_and_map_edges(get_giant_component(G1), get_giant_component(G2),edge_label='title')
    
    
    for i, G in enumerate([G1, G2]):
        G_giant = get_giant_component(G)

        #nx.write_graphml(G_giant, os.path.join(data_dir, f"top_{i}_subgraph1_giant_{root}.graphml"))
        pos = graphviz_layout(G_giant, prog='neato')  # Using 'neato' for improved layout

        #pos = nx.spring_layout(G, seed=42,k=0.15, iterations=100) 

        
        communities = greedy_modularity_communities(G_giant)
        degrees = G_giant.degree()
        node_sizes = [degrees[n]*500 for n in G_giant.nodes()]  # Scale node size

        
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(communities))))
        for community in communities:
            community_nodes = list(community)
            community_node_sizes = [degrees[n]*500 for n in community_nodes]  # Apply scaling for community nodes
            nx.draw_networkx_nodes(G_giant, pos, nodelist=community_nodes, node_size=community_node_sizes, node_color=next(colors), ax=axs[i], alpha=0.7)
        
        nx.draw_networkx_edges(G_giant, pos, ax=axs[i], alpha=0.5, width=2)
        
        # Preprocess edge labels to split long labels into multiple lines
        edge_labels = {edge: split_label_forgraph(label) for edge, label in nx.get_edge_attributes(G_giant, 'title').items()}
        nx.draw_networkx_edge_labels(G_giant, pos, edge_labels=edge_labels, ax=axs[i], font_size=8)

        # Preprocess node labels similarly
        node_labels = {node: split_label_forgraph(data.get('label', node)) for node, data in G_giant.nodes(data=True)}
        nx.draw_networkx_labels(G_giant, pos, labels=node_labels, ax=axs[i], font_size=10, font_color='black')

        axs[i].set_title(title1 if i == 0 else title2, fontsize=16)
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename, format='svg')
    plt.show()  # Close the plot figure window after saving to fil

    return node_mapping, edge_mapping


def find_and_save_isomorphic_subgraphs_with_communities(G1, G2, data_dir, root='graph', top_n=5, max_iter=100, min_avg_degree=2,
                                                       min_component_size=10):
    os.makedirs(data_dir, exist_ok=True)
    matcher = GraphMatcher(G1, G2)
    
    scores_subgraphs = []
    iter_count = 0  # Initialize iteration counter

    for iso_map in tqdm(matcher.subgraph_isomorphisms_iter()):
        if iter_count >= max_iter:
            break  # Exit the loop if the maximum number of iterations is reached
    
        subgraph1 = G1.subgraph(iso_map.keys())
        subgraph2 = G2.subgraph(iso_map.values())
        
        # Apply a connectivity filter based on average degree
        avg_degree1 = sum(dict(subgraph1.degree()).values()) / float(subgraph1.number_of_nodes())
        avg_degree2 = sum(dict(subgraph2.degree()).values()) / float(subgraph2.number_of_nodes())
        if avg_degree1 < min_avg_degree or avg_degree2 < min_avg_degree:
            continue  # Skip subgraphs with low connectivity
            
        largest_component1 = max(connected_components(subgraph1), key=len)
        largest_component2 = max(connected_components(subgraph2), key=len)
        if len(largest_component1) < min_component_size and len(largest_component2) < min_component_size:
            continue  # Skip this pair if neither subgraph has a large enough connected component


        iter_count += 1 #increment if succesfully identified higher degree cases
        communities1 = list(greedy_modularity_communities(subgraph1))
        communities2 = list(greedy_modularity_communities(subgraph2))
        score1 = modularity(subgraph1, communities1)
        score2 = modularity(subgraph2, communities2)
        avg_score = (score1 + score2) / 2
        scores_subgraphs.append((avg_score, subgraph1, subgraph2))

    
    scores_subgraphs.sort(reverse=True, key=lambda x: x[0])

    print ("Done, scores: ",scores_subgraphs[:top_n])  
    for i, (score, subgraph1, subgraph2) in enumerate(scores_subgraphs[:top_n], 1):
    
        nx.write_graphml(subgraph1, os.path.join(data_dir, f"top_{i}_subgraph1_{root}.graphml"))
        nx.write_graphml(subgraph2, os.path.join(data_dir, f"top_{i}_subgraph2_{root}.graphml"))

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        visualize_community_structure(subgraph1, f"Top {i} Subgraph 1 Communities (Score: {score:.2f})", ax=axs[0])
        visualize_community_structure(subgraph2, f"Top {i} Subgraph 2 Communities (Score: {score:.2f})", ax=axs[1])

        visualization_filename = os.path.join(data_dir, f"top_{i}_subgraph_communities_{root}.svg")
        plt.tight_layout()
        plt.savefig(visualization_filename, format="svg")
        plt.show ()
        plt.close()

        visualize_community_structure_in_giant_component(subgraph1, subgraph2, title1=f"Top {i} Subgraph 1 Giant Component", 
                                                         title2=f"Top {i} Subgraph 2 Giant Component", filename=f"{data_dir}/top_{i}_subgraph1-2_giant_{root}.svg",
                                                         data_dir=data_dir, root='graph')

        nx.write_graphml(get_giant_component(subgraph1), os.path.join(data_dir, f"top_{i}_subgraph1_GIANT_F_{root}.graphml"))
        nx.write_graphml(get_giant_component(subgraph2), os.path.join(data_dir, f"top_{i}_subgraph2_GIANT_F_{root}.graphml"))
    

    print(f"Top {top_n} isomorphic subgraphs processed and saved with community analysis.")

def return_giant_component_of_graph (G_new ):
    connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
    G_new = G_new.subgraph(connected_components[0]).copy()
   
    return G_new#, node_embeddings

import networkx as nx
def calculate_bridging_coefficient(G):
    # Calculate the degree for all nodes
    degrees = dict(G.degree())
    # Initialize a dictionary to hold the bridging coefficient of each node
    bridging_coefficient = {}
    
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        bridging_coefficient[node] = (1 / degrees[node]) * sum((1 / degrees[neighbor]) for neighbor in neighbors)
    
    return bridging_coefficient

def remove_top_n_bridging_centrality(G, top_N):
    # Calculate betweenness centrality for all nodes
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Calculate bridging coefficient for all nodes
    bridging_coefficient = calculate_bridging_coefficient(G)
    
    # Calculate bridging centrality for all nodes
    bridging_centrality = {node: betweenness_centrality[node] * bridging_coefficient[node] for node in G.nodes()}
    
    # Sort nodes by bridging centrality and select the top N nodes
    top_n_nodes = sorted(bridging_centrality, key=bridging_centrality.get, reverse=True)[:top_N]

    # Create a new graph without the top N nodes
    G_new = G.copy()
    G_new.remove_nodes_from(top_n_nodes)

    return G_new

def include_top_n_betweenness_centrality(G, top_N=5):
    # Calculate betweenness centrality for all nodes
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Sort nodes by betweenness centrality and select the top N nodes
    top_n_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:top_N]

    # Print the names and betweenness centrality of the top N nodes
    print ("############################################")
    print("Top N nodes by betweenness centrality:")
    
    for node in top_n_nodes:
        print(f"Node: {node}, Betweenness Centrality: {betweenness_centrality[node]}")
        
    # Create a new graph that includes only the top N nodes
    G_new = nx.Graph()
    G_new.add_nodes_from(top_n_nodes)
    # Add edges between these nodes if they existed in the original graph
    for node1 in top_n_nodes:
        for node2 in top_n_nodes:
            if G.has_edge(node1, node2):
                G_new.add_edge(node1, node2, **G[node1][node2])
    
    return G_new

# Define the function to calculate and add bridging centrality as a node attribute
def add_bridging_and_centrality_attributes(G):
    G_new=G.copy()
    # Calculate betweenness centrality for all nodes
    betweenness_centrality = nx.betweenness_centrality(G_new)
    
    # Calculate bridging coefficient for all nodes
    bridging_coefficient = calculate_bridging_coefficient(G_new)
    
    # Calculate bridging centrality for all nodes and add it as a node attribute
    for node in G_new.nodes():
        G_new.nodes[node]['bridging_centrality_nwx'] = betweenness_centrality[node] * bridging_coefficient[node]
        G_new.nodes[node]['bridging_coefficient_nwx'] =   bridging_coefficient[node]
        G_new.nodes[node]['betweenness_centrality_nwx'] =   betweenness_centrality[node]
        
    return G_new
    
# Define the function to save the graph with the new node attribute to a GraphML file
def save_graph_with_bridging_and_centrality_attributes(G, filename='graph_with_bridging_centrality.graphml'):
    G_new=add_bridging_and_centrality_attributes(G)  # Add the bridging centrality attribute to each node
    nx.write_graphml(G_new, filename)  # Save the graph to a GraphML file

def use_graph_and_reason_over_triples (path_graph, generate, 
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
    make_dir_if_needed(data_dir)
    task=inst_prepend+''

    join_strings = lambda strings: '\n'.join(strings)
    join_strings_newline = lambda strings: '\n'.join(strings)

    node_list=print_node_pairs_edge_title(path_graph)
    if N_limit != None:
        node_list=node_list[:N_limit]

    print ("Node list: ", node_list)

    if include_keywords_as_nodes:
        task=task+f"The following is a graph provided from an analysis of relationships between the concepts of {keyword_1} and {keyword_2}.\n\n"
    task=task+f"{inst_prepend}Consider this list of nodes and relations in a knowledge graph:\n\nFormat: node_1, relationship, node_2\n\nThe data is:\n\n{join_strings_newline( node_list)}\n\n"

    task=task+f"{inst_prepend}{instruction}"
    
    print (task)
    
    response=generate(system_prompt=system_prompt, #local_llm=local_llm,
         prompt=task, max_tokens=max_tokens, temperature=temperature)
    
    display(Markdown("**Response:** "+response ))

    return response ,  path_graph, shortest_path_length, fname, graph_GraphML