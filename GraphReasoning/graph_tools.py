import heapq

import copy
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
import community as community_louvain
import networkx as nx
import pandas as pd  # Assuming colors2Community returns a pandas DataFrame

import seaborn as sns
import re
from IPython.display import display, Markdown

import markdown2
import pdfkit

import time

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
  
# Function to generate embeddings
def generate_node_embeddings(graph, tokenizer, model):
    embeddings = {}
    for node in tqdm(graph.nodes()):
        inputs = tokenizer(str(node), return_tensors="pt")
        outputs = model(**inputs)
        embeddings[node] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

import pickle

def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings
 
def find_best_fitting_node(keyword, embeddings, tokenizer, model):
    inputs = tokenizer(keyword, return_tensors="pt")
    outputs = model(**inputs)
    keyword_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()  # Flatten to ensure 1-D
    
    # Calculate cosine similarity and find the best match
    best_node = None
    best_similarity = float('-inf')  # Initialize with negative infinity
    for node, embedding in embeddings.items():
        # Ensure embedding is 1-D
        embedding = embedding.flatten()  # Flatten to ensure 1-D
        similarity = 1 - cosine(keyword_embedding, embedding)  # Cosine similarity
        if similarity > best_similarity:
            best_similarity = similarity
            best_node = node
            
    return best_node, best_similarity

def find_best_fitting_node_list(keyword, embeddings, tokenizer, model, N_samples=5):
    inputs = tokenizer(keyword, return_tensors="pt")
    outputs = model(**inputs)
    keyword_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()  # Flatten to ensure 1-D
    
    # Initialize a min-heap
    min_heap = []
    heapq.heapify(min_heap)
    
    for node, embedding in embeddings.items():
        # Ensure embedding is 1-D
        embedding = embedding.flatten()  # Flatten to ensure 1-D
        similarity = 1 - cosine(keyword_embedding, embedding)  # Cosine similarity
        
        # If the heap is smaller than N_samples, just add the current node and similarity
        if len(min_heap) < N_samples:
            heapq.heappush(min_heap, (similarity, node))
        else:
            # If the current similarity is greater than the smallest similarity in the heap
            if similarity > min_heap[0][0]:
                heapq.heappop(min_heap)  # Remove the smallest
                heapq.heappush(min_heap, (similarity, node))  # Add the current node and similarity
                
    # Convert the min-heap to a sorted list in descending order of similarity
    best_nodes = sorted(min_heap, key=lambda x: -x[0])
    
    # Return a list of tuples (node, similarity)
    return [(node, similarity) for similarity, node in best_nodes]


# Example usage
def visualize_embeddings_2d(embeddings , data_dir='./'):
    # Generate embeddings
    #embeddings = generate_node_embeddings(graph, tokenizer, model)
    
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
    for i, node_id in enumerate(node_ids):
        plt.text(vectors_2d[i, 0], vectors_2d[i, 1], str(node_id), fontsize=9)
    plt.title('Node Embeddings Visualization')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.savefig(f'{data_dir}/node_embeddings_2d.svg')  # Save the figure as SVG
    plt.show()


def visualize_embeddings_2d_notext(embeddings, n_clusters=3, data_dir='./'):
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    labels = kmeans.labels_
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, alpha=0.5, cmap='viridis')
    plt.title('Node Embeddings Visualization')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters.svg')  # Save the figure as SVG
    plt.show()


def visualize_embeddings_2d_pretty(embeddings, n_clusters=3,  data_dir='./'):
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    labels = kmeans.labels_
    
    # Count the number of points in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')  # Set seaborn style for prettier plots
    
    # Use seaborn's color palette and matplotlib's scatter plot
    palette = sns.color_palette("hsv", n_clusters)  # Use a different color palette
    for cluster in range(n_clusters):
        cluster_points = vectors_2d[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster} (n={cluster_counts[cluster]})', alpha=0.7, edgecolors='w', s=100, cmap=palette)
    
    plt.title('Node Embeddings Visualization with Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(scatterpoints=1)  # Add a legend to show cluster labels and counts
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters_pretty.svg')  # Save the figure as SVG
    plt.show()
    
    # Optionally print the counts for each cluster
    for cluster, count in cluster_counts.items():
        print(f'Cluster {cluster}: {count} items')

from scipy.spatial.distance import cdist

def visualize_embeddings_2d_pretty_and_sample(embeddings, n_clusters=3, n_samples=5, data_dir='./',
                                             alpha=0.7, edgecolors='none', s=50,):
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    labels = kmeans.labels_
    
    # Count the number of points in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')  # Set seaborn style for prettier plots
    palette = sns.color_palette("hsv", n_clusters)
    for cluster in range(n_clusters):
        cluster_points = vectors_2d[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster} (n={cluster_counts[cluster]})'
                    , alpha=alpha, edgecolors=edgecolors, s=s, cmap=palette,#alpha=0.7, edgecolors='w', s=100, cmap=palette)
                   )
    
    plt.title('Node Embeddings Visualization with Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(scatterpoints=1)
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters_pretty.svg')
    plt.show()
    
    # Output N_sample terms from the center of each cluster
    centroids = kmeans.cluster_centers_
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_vectors = vectors[cluster_indices]
        cluster_node_ids = np.array(node_ids)[cluster_indices]
        
        # Calculate distances of points in this cluster to the centroid
        distances = cdist(cluster_vectors, [centroids[cluster]], 'euclidean').flatten()
        
        # Get indices of N_samples closest points
        closest_indices = np.argsort(distances)[:n_samples]
        closest_node_ids = cluster_node_ids[closest_indices]
        
        print(f'Cluster {cluster}: {len(cluster_vectors)} items')
        print(f'Closest {n_samples} node IDs to centroid:', closest_node_ids)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def visualize_embeddings_with_gmm_density_voronoi_and_print_top_samples(embeddings, n_clusters=5, top_n=3, data_dir='./',s=50):
    # Extract the embedding vectors
    descriptions = list(embeddings.keys())
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(vectors_2d)
    labels = gmm.predict(vectors_2d)
    
    # Generate Voronoi regions
    vor = Voronoi(gmm.means_)
    
    # Plotting
    plt.figure(figsize=(10, 10))
    # Plot Voronoi diagram
    voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors='black', line_width=1, line_alpha=0.7, point_size=2)
    
    # Color points based on their cluster
    for i in range(n_clusters):
        plt.scatter(vectors_2d[labels == i, 0], vectors_2d[labels == i, 1], s=s, label=f'Cluster {i}')
    
    plt.title('Embedding Vectors with GMM Density and Voronoi Tessellation')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters_voronoi.svg')
    
    plt.show()
    # Print top-ranked sample texts
    for i in range(n_clusters):
        cluster_center = gmm.means_[i]
        cluster_points = vectors_2d[labels == i]
        
        distances = euclidean_distances(cluster_points, [cluster_center])
        distances = distances.flatten()
        
        closest_indices = np.argsort(distances)[:top_n]
        
        print(f"\nTop {top_n} closest samples to the center of Cluster {i}:")
        for idx in closest_indices:
            original_idx = np.where(labels == i)[0][idx]
            desc = descriptions[original_idx]
            print(f"- Description: {desc}, Distance: {distances[idx]:.2f}")


def graph_statistics_and_plots(G, data_dir='./'):
    # Calculate statistics
    degrees = [degree for node, degree in G.degree()]
    degree_distribution = np.bincount(degrees)
    average_degree = np.mean(degrees)
    clustering_coefficients = nx.clustering(G)
    average_clustering_coefficient = nx.average_clustering(G)
    triangles = sum(nx.triangles(G).values()) / 3
    connected_components = nx.number_connected_components(G)
    density = nx.density(G)
    
    # Diameter and Average Path Length (for connected graphs or components)
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        average_path_length = nx.average_shortest_path_length(G)
    else:
        diameter = "Graph not connected"
        component_lengths = [nx.average_shortest_path_length(G.subgraph(c)) for c in nx.connected_components(G)]
        average_path_length = np.mean(component_lengths)
    
    # Plot Degree Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), alpha=0.75, color='blue')
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.savefig(f'{data_dir}/degree_distribution.svg')
    #plt.close()
    plt.show()
    
    # Plot Clustering Coefficient Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(list(clustering_coefficients.values()), bins=10, alpha=0.75, color='green')
    plt.title('Clustering Coefficient Distribution')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.savefig(f'{data_dir}/clustering_coefficient_distribution.svg')
    plt.show()
    #plt.close()
    
    statistics = {
        'Degree Distribution': degree_distribution,
        'Average Degree': average_degree,
        'Clustering Coefficients': clustering_coefficients,
        'Average Clustering Coefficient': average_clustering_coefficient,
        'Number of Triangles': triangles,
        'Connected Components': connected_components,
        'Diameter': diameter,
        'Density': density,
        'Average Path Length': average_path_length,
    }
    
    return statistics
 
def graph_statistics_and_plots_for_large_graphs(G, data_dir='./', include_centrality=False,
                                               make_graph_plot=False,root='graph'):
    # Basic statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = [degree for node, degree in G.degree()]
    log_degrees = np.log1p(degrees)  # Using log1p for a better handle on zero degrees
    #degree_distribution = np.bincount(degrees)
    average_degree = np.mean(degrees)
    density = nx.density(G)
    connected_components = nx.number_connected_components(G)
    
    # Centrality measures
    if include_centrality:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Community detection with Louvain method
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))

    # Plotting
    # Degree Distribution on a log-log scale
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(log_degrees, bins=50, alpha=0.75, color='blue', log=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Log-Log Degree Distribution')
    plt.xlabel('Degree (log)')
    plt.ylabel('Frequency (log)')
    plt.savefig(f'{data_dir}/loglog_degree_distribution_{root}.svg')
    plt.close()

    if make_graph_plot:
        
        # Additional Plots
        # Plot community structure
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G)  # for better visualization
        cmap = plt.get_cmap('viridis')
        nx.draw_networkx(G, pos, node_color=list(partition.values()), node_size=20, cmap=cmap, with_labels=False)
        plt.title('Community Structure')
        plt.savefig(f'{data_dir}/community_structure_{root}.svg')
        plt.close()

    # Save statistics
    statistics = {
        'Number of Nodes': num_nodes,
        'Number of Edges': num_edges,
        'Average Degree': average_degree,
        'Density': density,
        'Connected Components': connected_components,
        'Number of Communities': num_communities,
        # Centrality measures could be added here as well, but they are often better analyzed separately due to their detailed nature
    }
    if include_centrality:
        centrality = {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'eigenvector_centrality': eigenvector_centrality,
        }
    else:
        centrality=None
 
    
    return statistics, include_centrality

## Now add these colors to communities and make another dataframe
def colors2Community(communities) -> pd.DataFrame:
    ## Define a color palette
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


 
def graph_Louvain (G, 
                  graph_GraphML=None, palette = "hls"):
    # Assuming G is your graph and data_dir is defined
    
    # Compute the best partition using the Louvain algorithm
    partition = community_louvain.best_partition(G)
    
    # Organize nodes into communities based on the Louvain partition
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    
    communities_list = list(communities.values())
    print("Number of Communities =", len(communities_list))
    print("Communities: ", communities_list)
    
    # Assuming colors2Community can work with the communities_list format
    colors = colors2Community(communities_list)
    print("Colors: ", colors)
    
    # Assign attributes to nodes based on their community membership
    for index, row in colors.iterrows():
        node = row['node']
        G.nodes[node]['group'] = row['group']
        G.nodes[node]['color'] = row['color']
        G.nodes[node]['size'] = G.degree[node]
    
    print("Done, assigned colors and groups...")
    
    # Write the graph with community information to a GraphML file
    if graph_GraphML != None:
        try:
            nx.write_graphml(G, graph_GraphML)
    
            print("Written GraphML.")

        except:
            print ("Error saving GraphML file.")
    return G
    
def save_graph (G, 
                  graph_GraphML=None, ):
    if graph_GraphML != None:
        nx.write_graphml(G, graph_GraphML)
    
        print("Written GraphML")
    else:
        print("Error, no file name provided.")
    return 

def update_node_embeddings(embeddings, graph_new, tokenizer, model, remove_embeddings_for_nodes_no_longer_in_graph=True,
                          verbatim=False):
    """
    Update embeddings for new nodes in an updated graph, ensuring that the original embeddings are not altered.

    Args:
    - embeddings (dict): Existing node embeddings.
    - graph_new: The updated graph object.
    - tokenizer: Tokenizer object to tokenize node names.
    - model: Model object to generate embeddings.

    Returns:
    - Updated embeddings dictionary with embeddings for new nodes, without altering the original embeddings.
    """
    # Create a deep copy of the original embeddings
    embeddings_updated = copy.deepcopy(embeddings)
    
    # Iterate through new graph nodes
    for node in tqdm(graph_new.nodes()):
        # Check if the node already has an embedding in the copied dictionary
        if node not in embeddings_updated:
            if verbatim:
                print(f"Generating embedding for new node: {node}")
            inputs = tokenizer(node, return_tensors="pt")
            outputs = model(**inputs)
            # Update the copied embeddings dictionary with the new node's embedding
            embeddings_updated[node] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    if remove_embeddings_for_nodes_no_longer_in_graph:
        # Remove embeddings for nodes that no longer exist in the graph from the copied dictionary
        nodes_in_graph = set(graph_new.nodes())
        for node in list(embeddings_updated):
            if node not in nodes_in_graph:
                if verbatim:
                    print(f"Removing embedding for node no longer in graph: {node}")
                del embeddings_updated[node]

    return embeddings_updated

def remove_small_fragents (G_new, size_threshold):
    if size_threshold >0:
        
        # Find all connected components, returned as sets of nodes
        components = list(nx.connected_components(G_new))
        
        # Iterate through components and remove those smaller than the threshold
        for component in components:
            if len(component) < size_threshold:
                # Remove the nodes in small components
                G_new.remove_nodes_from(component)
    return G_new


def simplify_node_name_with_llm(node_name, generate, max_tokens=2048, temperature=0.3):
    # Generate a prompt for the LLM to simplify or describe the node name
    system_prompt='You are an ontological graph maker. You carefully rename nodes in complex networks.'
    prompt = f"Provide a simplified, more descriptive name for a network node named '{node_name}' that reflects its importance or role within a network."
   
    # Assuming 'generate' is a function that calls the LLM with the given prompt
    #simplified_name = generate(system_prompt=system_prompt, prompt)
    simplified_name = generate(system_prompt=system_prompt, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
   
    return simplified_name

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def simplify_graph_simple(graph_, node_embeddings, tokenizer, model, similarity_threshold=0.9, use_llm=False,
                  data_dir_output='./',
                  graph_root='simple_graph', verbatim=False,max_tokens=2048, temperature=0.3,generate=None,
                  ):
    graph = graph_.copy()
    nodes = list(node_embeddings.keys())
    embeddings_matrix = np.array([node_embeddings[node].flatten() for node in nodes])

    similarity_matrix = cosine_similarity(embeddings_matrix)
    to_merge = np.where(similarity_matrix > similarity_threshold)

    node_mapping = {}
    nodes_to_recalculate = set()
    for i, j in tqdm(zip(*to_merge)):
        if i != j:  # ignore self-similarity
            node_i, node_j = nodes[i], nodes[j]
            if graph.degree(node_i) >= graph.degree(node_j):
                node_to_keep, node_to_merge = node_i, node_j
            else:
                node_to_keep, node_to_merge = node_j, node_i
            if verbatim:
                print ("node to keep and merge: ",  node_to_keep,"<--",  node_to_merge)
            # Optionally use LLM to generate a simplified or more descriptive name
            if use_llm:
                original_node_to_keep = node_to_keep
                node_to_keep = simplify_node_name_with_llm(node_to_keep, generate, max_tokens=max_tokens, temperature=temperature)
                # Add the original and new node names to the list for recalculation
                nodes_to_recalculate.add(original_node_to_keep)
                nodes_to_recalculate.add(node_to_keep)
            
            node_mapping[node_to_merge] = node_to_keep

    new_graph = nx.relabel_nodes(graph, node_mapping, copy=True)

    # Recalculate embeddings for nodes that have been merged or renamed
    recalculated_embeddings = regenerate_node_embeddings(new_graph, nodes_to_recalculate, tokenizer, model)
    
    # Update the embeddings dictionary with the recalculated embeddings
    updated_embeddings = {**node_embeddings, **recalculated_embeddings}

    # Remove embeddings for nodes that no longer exist
    for node in node_mapping.keys():
        if node in updated_embeddings:
            del updated_embeddings[node]

    graph_GraphML=  f'{data_dir_output}/{graph_root}_graphML_simplified.graphml'  #  f'{data_dir}/resulting_graph.graphml',
        #print (".")
    nx.write_graphml(new_graph, graph_GraphML)
    
    return new_graph, updated_embeddings

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
from powerlaw import Fit

# Assuming regenerate_node_embeddings is defined as provided earlier

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def simplify_node_name_with_llm(node_name, max_tokens, temperature):
    # This is a placeholder for the actual function that uses a language model
    # to generate a simplified or more descriptive node name.
    return node_name  

def regenerate_node_embeddings(graph, nodes_to_recalculate, tokenizer, model):
    """
    Regenerate embeddings for specific nodes.
    """
    new_embeddings = {}
    for node in tqdm(nodes_to_recalculate):
        inputs = tokenizer(node, return_tensors="pt")
        outputs = model(**inputs)
        new_embeddings[node] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return new_embeddings
    
def simplify_graph(graph_, node_embeddings, tokenizer, model, similarity_threshold=0.9, use_llm=False,
                   data_dir_output='./', graph_root='simple_graph', verbatim=False, max_tokens=2048, 
                   temperature=0.3, generate=None):
    """
    Simplifies a graph by merging similar nodes and optionally renaming them using a language model.
    """

    graph = graph_.copy()
    
    nodes = list(node_embeddings.keys())
    embeddings_matrix = np.array([node_embeddings[node].flatten() for node in nodes])

    similarity_matrix = cosine_similarity(embeddings_matrix)
    to_merge = np.where(similarity_matrix > similarity_threshold)

    node_mapping = {}
    nodes_to_recalculate = set()
    merged_nodes = set()  # Keep track of nodes that have been merged
    if verbatim:
        print("Start...")
    for i, j in tqdm(zip(*to_merge), total=len(to_merge[0])):
        if i != j and nodes[i] not in merged_nodes and nodes[j] not in merged_nodes:  # Check for duplicates
            node_i, node_j = nodes[i], nodes[j]
            
            try:
                if graph.degree(node_i) >= graph.degree(node_j):
                #if graph.degree[node_i] >= graph.degree[node_j]:
                    node_to_keep, node_to_merge = node_i, node_j
                else:
                    node_to_keep, node_to_merge = node_j, node_i
    
                if verbatim:
                    print("Node to keep and merge:", node_to_keep, "<--", node_to_merge)
    
                #if use_llm and node_to_keep in nodes_to_recalculate:
                #    node_to_keep = simplify_node_name_with_llm(node_to_keep, max_tokens=max_tokens, temperature=temperature)
    
                node_mapping[node_to_merge] = node_to_keep
                nodes_to_recalculate.add(node_to_keep)
                merged_nodes.add(node_to_merge)  # Mark the merged node to avoid duplicate handling
            except:
                print (end="")
    if verbatim:
        print ("Now relabel. ")
    # Create the simplified graph by relabeling nodes.
    new_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    if verbatim:
        print ("New graph generated, nodes relabled. ")
    # Recalculate embeddings for nodes that have been merged or renamed.
    recalculated_embeddings = regenerate_node_embeddings(new_graph, nodes_to_recalculate, tokenizer, model)
    if verbatim:
        print ("Relcaulated embeddings... ")
    # Update the embeddings dictionary with the recalculated embeddings.
    updated_embeddings = {**node_embeddings, **recalculated_embeddings}

    # Remove embeddings for nodes that no longer exist in the graph.
    for node in merged_nodes:
        updated_embeddings.pop(node, None)
    if verbatim:
        print ("Now save graph... ")

    # Save the simplified graph to a file.
    graph_path = f'{data_dir_output}/{graph_root}_graphML_simplified.graphml'
    nx.write_graphml(new_graph, graph_path)

    if verbatim:
        print(f"Graph simplified and saved to {graph_path}")

    return new_graph, updated_embeddings

def make_HTML (G,data_dir='./', graph_root='graph_root'):

    net = Network(
            #notebook=False,
            notebook=True,
            # bgcolor="#1a1a1a",
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            # font_color="#cccccc",
            filter_menu=False,
        )
        
    net.from_nx(G)
    # net.repulsion(node_distance=150, spring_length=400)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
    # net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)
    
    #net.show_buttons(filter_=["physics"])
    net.show_buttons()
    
    #net.show(graph_output_directory, notebook=False)
    graph_HTML= f'{data_dir}/{graph_root}_graphHTML.html'
    
    net.show(graph_HTML, #notebook=True
            )

    return graph_HTML

def return_giant_component_of_graph (G_new ):
    connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
    G_new = G_new.subgraph(connected_components[0]).copy()
    return G_new 
    
def return_giant_component_G_and_embeddings (G_new, node_embeddings):
    connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
    G_new = G_new.subgraph(connected_components[0]).copy()
    node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)
    return G_new, node_embeddings

def extract_number(filename):
    # This function extracts numbers from a filename and converts them to an integer.
    # It finds all sequences of digits in the filename and returns the first one as an integer.
    # If no number is found, it returns -1.
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else -1
 
def get_list_of_graphs_and_chunks (graph_q='graph_*_graph_clean.csv',  chunk_q='graph_*_chunks_clean.csv', data_dir='./',verbatim=False):
    graph_pattern = os.path.join(data_dir, graph_q)
    chunk_pattern = os.path.join(data_dir, chunk_q)
    
    # Use glob to find all files matching the patterns
    graph_files = glob.glob(graph_pattern)
    chunk_files = glob.glob(chunk_pattern)
    
    # Sort the files using the custom key function
    graph_file_list = sorted(graph_files, key=extract_number)
    chunk_file_list = sorted(chunk_files, key=extract_number)

    if verbatim:
        # Print the lists to verify
        print ('\n'.join(graph_file_list[:10]), '\n\n', '\n'.join(chunk_file_list[:10]),'\n')
        
        print('# graph files:', len (graph_file_list))
        print('# chunk files:', len (chunk_file_list))
    
    return graph_file_list, chunk_file_list

def print_graph_nodes_with_texts(G, separator="; ", N=64):
    """
    Prints out each node in the graph along with the associated texts, concatenated into a single string.

    Parameters:
    - G: A NetworkX graph object where each node has a 'texts' attribute containing a list of texts.
    - separator: A string separator used to join texts. Default is "; ".
    """
    print("Graph Nodes and Their Associated Texts (Concatenated):")
    for node, data in G.nodes(data=True):
        texts = data.get('texts', [])
        concatenated_texts = separator.join(texts)
        print(f"Node: {node}, Texts: {concatenated_texts[:N]}")      
       
def print_graph_nodes (G, separator="; ", N=64):
    """
    Prints out each node in the graph along with the associated texts, concatenated into a single string.

    Parameters:
    - G: A NetworkX graph object where each node has a 'texts' attribute containing a list of texts.
    - separator: A string separator used to join texts. Default is "; ".
    """
    i=0
    print("Graph Nodes and Their Associated Texts (Concatenated):")
    for node in G.nodes :
        print(f"Node {i}: {node}")  
        i=i+1
def get_text_associated_with_node(G, node_identifier ='bone', ):
        
    # Accessing and printing the 'texts' attribute for the node
    if 'texts' in G.nodes[node_identifier]:
        texts = G.nodes[node_identifier]['texts']
        concatenated_texts = "; ".join(texts)  # Assuming you want to concatenate the texts
        print(f"Texts associated with node '{node_identifier}': {concatenated_texts}")
    else:
        print(f"No 'texts' attribute found for node {node_identifier}")
        concatenated_texts=''
    return concatenated_texts 

import networkx as nx
import json
from copy import deepcopy
from tqdm import tqdm

def save_graph_with_text_as_JSON(G_or, data_dir='./', graph_name='my_graph.graphml'):
    G = deepcopy(G_or)

    # Ensure correct path joining
    import os
    fname = os.path.join(data_dir, graph_name)

    for _, data in tqdm(G.nodes(data=True)):
        for key in data:
            if isinstance(data[key], (list, dict, set, tuple)):  # Extend this as needed
                data[key] = json.dumps(data[key])

    for _, _, data in tqdm(G.edges(data=True)):
        for key in data:
            if isinstance(data[key], (list, dict, set, tuple)):  # Extend this as needed
                data[key] = json.dumps(data[key])

    nx.write_graphml(G, fname)
    return fname

def load_graph_with_text_as_JSON(data_dir='./', graph_name='my_graph.graphml'):
    # Ensure correct path joining
    import os
    fname = os.path.join(data_dir, graph_name)

    G = nx.read_graphml(fname)

    for node, data in tqdm(G.nodes(data=True)):
        for key, value in data.items():
            if isinstance(value, str):  # Only attempt to deserialize strings
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass  # If the value is not a valid JSON string, do nothing

    for _, _, data in tqdm(G.edges(data=True)):
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass

    return G

from copy import deepcopy
import networkx as nx
from tqdm import tqdm
import os

def save_graph_without_text(G_or, data_dir='./', graph_name='my_graph.graphml'):
    G = deepcopy(G_or)

    # Process nodes: remove 'texts' attribute and convert others to string
    for _, data in tqdm(G.nodes(data=True), desc="Processing nodes"):
        if 'texts' in data:
            del data['texts']  # Remove the 'texts' attribute
        # Convert all other attributes to strings
        for key in data:
            data[key] = str(data[key])

    # Process edges: similar approach, remove 'texts' and convert attributes
    for i, (_, _, data) in enumerate(tqdm(G.edges(data=True), desc="Processing edges")):
    #for _, _, data in tqdm(G.edges(data=True), desc="Processing edges"):
        data['id'] = str(i)  # Assign a unique ID
        if 'texts' in data:
            del data['texts']  # Remove the 'texts' attribute
        # Convert all other attributes to strings
        for key in data:
            data[key] = str(data[key])
    
    # Ensure correct directory path and file name handling
    fname = os.path.join(data_dir, graph_name)
    
    # Save the graph to a GraphML file
    nx.write_graphml(G, fname, edge_id_from_attribute='id')
    return fname

def print_nodes_and_labels (G, N=10):
    # Printing out the first 10 nodes
    ch_list=[]
    
    print("First 10 nodes:")
    for node in list(G.nodes())[:10]:
        print(node)
    
    print("\nFirst 10 edges with titles:")
    for (node1, node2, data) in list(G.edges(data=True))[:10]:
        edge_title = data.get('title')  # Replace 'title' with the attribute key you're interested in
        ch=f"Node labels: ({node1}, {node2}) - Title: {edge_title}"
        ch_list.append (ch)
        
        print (ch)
        

    return ch_list

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import json
from tqdm import tqdm
import pandas as pd
import networkx as nx
import os 
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import json

def make_graph_from_text_withtext(graph_file_list, chunk_file_list,
                                  include_contextual_proximity=False,
                                  graph_root='graph_root',
                                  repeat_refine=0, verbatim=False,
                                  data_dir='./data_output_KG/',
                                  save_PDF=False, save_HTML=True, N_max=10,
                                  idx_start=0):
    """
    Constructs a graph from text data, ensuring edge labels do not incorrectly include node names.
    """

    # Initialize an empty DataFrame to store all texts
    all_texts_df = pd.DataFrame()

    # Initialize an empty graph
    G_total = nx.Graph()

    for idx in tqdm(range(idx_start, min(len(graph_file_list), N_max)), desc="Processing graphs"):
        try:
            # Load graph and chunk data
            graph_df = pd.read_csv(graph_file_list[idx])
            text_df = pd.read_csv(chunk_file_list[idx])
            
            # Append the current text_df to the all_texts_df
            all_texts_df = pd.concat([all_texts_df, text_df], ignore_index=True)
    
            # Clean and aggregate the graph data
            graph_df.replace("", np.nan, inplace=True)
            graph_df.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
            graph_df['count'] = 4  # Example fixed count, adjust as necessary
            
            # Aggregate edges and combine attributes
            graph_df = (graph_df.groupby(["node_1", "node_2"])
                        .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
                        .reset_index())
            
            if verbatim:
                print("Shape of graph DataFrame: ", graph_df.shape)
    
            # Add edges to the graph
            for _, row in graph_df.iterrows():
                G_total.add_edge(row['node_1'], row['node_2'], chunk_id=row['chunk_id'],
                                 title=row['edge'], weight=row['count'] / 4)
    
        except Exception as e:
            print(f"Error in graph generation for idx={idx}: {e}")
   
    # Ensure no duplicate chunk_id entries
    all_texts_df = all_texts_df.drop_duplicates(subset=['chunk_id'])
    
    # Map chunk_id to text
    chunk_id_to_text = pd.Series(all_texts_df.text.values, index=all_texts_df.chunk_id).to_dict()

    # Initialize node texts collection
    node_texts = {node: set() for node in G_total.nodes()}

    # Associate texts with nodes based on edges
    for (node1, node2, data) in tqdm(G_total.edges(data=True), desc="Mapping texts to nodes"):
        chunk_ids = data.get('chunk_id', '').split(',')
        for chunk_id in chunk_ids:
            text = chunk_id_to_text.get(chunk_id, "")
            if text:  # If text is found for the chunk_id
                node_texts[node1].add(text)
                node_texts[node2].add(text)

    # Update nodes with their texts
    for node, texts in node_texts.items():
        G_total.nodes[node]['texts'] = list(texts)  # Convert from set to list

    return G_total
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def regenerate_node_embeddings(graph, nodes_to_recalculate, tokenizer, model):
    """
    Regenerate embeddings for specific nodes.
    """
    new_embeddings = {}
    for node in tqdm(nodes_to_recalculate):
        inputs = tokenizer(node, return_tensors="pt")
        outputs = model(**inputs)
        new_embeddings[node] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return new_embeddings
    
def simplify_graph_with_text(graph_, node_embeddings, tokenizer, model, similarity_threshold=0.9, use_llm=False,
                   data_dir_output='./', graph_root='simple_graph', verbatim=False, max_tokens=2048, 
                   temperature=0.3, generate=None):
    """
    Simplifies a graph by merging similar nodes and optionally renaming them using a language model.
    Also, merges 'texts' node attribute ensuring no duplicates.
    """

    graph = deepcopy(graph_)
    
    nodes = list(node_embeddings.keys())
    embeddings_matrix = np.array([node_embeddings[node].flatten() for node in nodes])

    similarity_matrix = cosine_similarity(embeddings_matrix)
    to_merge = np.where(similarity_matrix > similarity_threshold)

    node_mapping = {}
    nodes_to_recalculate = set()
    merged_nodes = set()  # Keep track of nodes that have been merged
    if verbatim:
        print("Start...")
    for i, j in tqdm(zip(*to_merge), total=len(to_merge[0])):
        if i != j and nodes[i] not in merged_nodes and nodes[j] not in merged_nodes:  # Check for duplicates
            node_i, node_j = nodes[i], nodes[j]
            
            try:
                if graph.degree(node_i) >= graph.degree(node_j):
                    node_to_keep, node_to_merge = node_i, node_j
                else:
                    node_to_keep, node_to_merge = node_j, node_i
    
                # Handle 'texts' attribute by merging and removing duplicates
                texts_to_keep = set(graph.nodes[node_to_keep].get('texts', []))
                texts_to_merge = set(graph.nodes[node_to_merge].get('texts', []))
                merged_texts = list(texts_to_keep.union(texts_to_merge))
                graph.nodes[node_to_keep]['texts'] = merged_texts
    
                if verbatim:
                    print("Node to keep and merge:", node_to_keep, "<--", node_to_merge)
    
                node_mapping[node_to_merge] = node_to_keep
                nodes_to_recalculate.add(node_to_keep)
                merged_nodes.add(node_to_merge)  # Mark the merged node to avoid duplicate handling
            except Exception as e:
                print("Error during merging:", e)
    if verbatim:
        print ("Now relabel. ")
    # Create the simplified graph by relabeling nodes.
    new_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    if verbatim:
        print ("New graph generated, nodes relabled. ")
    # Recalculate embeddings for nodes that have been merged or renamed.
    recalculated_embeddings = regenerate_node_embeddings(new_graph, nodes_to_recalculate, tokenizer, model)
    if verbatim:
        print ("Relcaulated embeddings... ")
    # Update the embeddings dictionary with the recalculated embeddings.
    updated_embeddings = {**node_embeddings, **recalculated_embeddings}
    if verbatim:
        print ("Done recalculate embeddings... ")
    
    # Remove embeddings for nodes that no longer exist in the graph.
    for node in merged_nodes:
        updated_embeddings.pop(node, None)
    if verbatim:
        print ("Now save graph... ")

    # Save the simplified graph to a file.
    graph_path = f'{graph_root}_graphML_simplified_JSON.graphml'
    save_graph_with_text_as_JSON (new_graph, data_dir=data_dir_output, graph_name=graph_path)
    
    if verbatim:
        print(f"Graph simplified and saved to {graph_path}")

    return new_graph, updated_embeddings
