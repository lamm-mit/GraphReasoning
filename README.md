# GraphReasoning: Scientific Discovery through Knowledge Extraction and Multimodal Graph-based Representation and Reasoning

Markus J. Buehler, MIT, 2024
mbuehler@MIT.EDU

Leveraging generative Artificial Intelligence (AI), we have transformed a dataset comprising 1,000 scientific papers into an ontological knowledge graph. Through an in-depth structural analysis, we have calculated node degrees, identified communities and connectivities, and evaluated clustering coefficients and betweenness centrality of pivotal nodes, uncovering fascinating knowledge architectures. The graph has an inherently scale-free nature, is highly connected, and can be used for graph reasoning by taking advantage of transitive and isomorphic properties that reveal unprecedented interdisciplinary relationships that can be used to answer queries, identify gaps in knowledge, propose never-before-seen material designs, and predict material behaviors. We compute deep node embeddings for combinatorial node similarity ranking for use in a path sampling strategy links dissimilar concepts that have previously not been related. One comparison revealed structural parallels between biological materials and Beethoven's 9th Symphony, highlighting shared patterns of complexity through isomorphic mapping. In another example, the algorithm proposed a hierarchical mycelium-based composite based on integrating path sampling with principles extracted from Kandinsky's 'Composition VII' painting. The resulting material integrates an innovative set of concepts that include a balance of chaos/order, adjustable porosity, mechanical strength, and complex patterned chemical functionalization. We uncover other isomorphisms across science, technology and art, revealing a nuanced ontology of immanence that reveal a context-dependent heterarchical interplay of constituents. Graph-based generative AI achieves a far higher degree of novelty, explorative capacity, and technical detail, than conventional approaches and establishes a widely useful framework for innovation by revealing hidden connections.
This library provides all codes and libraries used in the paper: https://arxiv.org/abs/2403.11996

![image](https://github.com/lamm-mit/GraphReasoning/assets/101393859/3baa3752-8222-4857-a64c-c046693d6315)

# Installation and Examples

Install directly from GitHub:
```
pip install git+https://github.com/lamm-mit/GraphReasoning
```
Or, editable:
```
pip install -e git+https://github.com/lamm-mit/GraphReasoning.git#egg=GraphReasoning
```
Install X-LoRA, if needed:
```
pip install git+https://github.com/EricLBuehler/xlora.git
```
You may need wkhtmltopdf for the multi-agent model:
```
sudo apt-get install wkhtmltopdf
```
If you plan to use llama.cpp, install using:
```
CMAKE_ARGS="-DLLAMA_CUBLAS=on " pip install  'git+https://github.com/abetlen/llama-cpp-python.git#egg=llama-cpp-python[server]' --force-reinstall --upgrade --no-cache-dir
```
Model weights and other data: 

[lamm-mit/GraphReasoning
](https://huggingface.co/lamm-mit/GraphReasoning/tree/main)

Graph file:
```
from huggingface_hub import hf_hub_download
data_dir='./GRAPHDATA/'    
graph_name='BioGraph.graphml'
filename = f"{data_dir}/{graph_name}"
file_path = hf_hub_download(repo_id=repository_id, filename=filename,  local_dir='./')
```
Embeddings: 
```
from huggingface_hub import hf_hub_download
data_dir='./GRAPHDATA/'    
embedding_file='BioGraph_embeddings_ge-large-en-v1.5.pkl'
filename = f"{data_dir}/{embedding_file}"
file_path = hf_hub_download(repo_id=repository_id, filename=filename, local_dir='./')
```
Example:
```
from transformers import AutoTokenizer, AutoModel

from GraphReasoning import *

embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, ) 
embedding_model = AutoModel.from_pretrained(tokenizer_model, ) 

data_dir_output='./GRAPHDATA_OUTPUT/'
make_dir_if_needed(data_dir_output)

data_dir='./GRAPHDATA/'    

graph_name=f'{data_dir}/{graph_name}'
G = nx.read_graphml(graph_name)
node_embeddings = load_embeddings(f'{data_dir}/{embedding_file}')

visualize_embeddings_2d_pretty_and_sample(node_embeddings,
                                            n_clusters=10, n_samples=10,
                                            data_dir=data_dir_output, alpha=.7)

describe_communities_with_plots_complex(G, N=6, data_dir=data_dir_output)
```
Analyze graph and extract information:
```
find_best_fitting_node_list("copper", node_embeddings, embedding_tokenizer, embedding_model, 5)
```
Find path:
```
(best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML=find_path( G, node_embeddings,
                                embedding_tokenizer, embedding_model , second_hop=False, data_dir=data_dir_output,
                                  keyword_1 = "copper", keyword_2 = "silk",
                                      similarity_fit_ID_node_1=0, similarity_fit_ID_node_2=0,
                                       )
path_list, path_string=print_path_with_edges_as_list(G , path)
path_list, path_string, path
```

# Reference

```LaTeX
@misc{Buehler2024AcceleratingDiscoveryGraphReasoning,
    author = {Buehler, Markus J.},
    title = {Accelerating Scientific Discovery with Generative Knowledge Extraction, Graph-Based Representation, and Multimodal Intelligent Graph Reasoning},
    year = {2024},
    eprint = {2403.11996},
    archivePrefix = {arXiv},
    doi = {10.48550/arXiv.2403.11996},
    url = {https://doi.org/10.48550/arXiv.2403.11996}
}
```

# API Documentation for Graph Analysis and Reasoning Code

## Table of Contents

1. [Introduction](#introduction)
2. [Graph Analysis](#graph-analysis)
   - [Shortest Path Functions](#shortest-path-functions)
   - [Path Finding and Reasoning](#path-finding-and-reasoning)
   - [Community Detection and Analysis](#community-detection-and-analysis)
   - [Scale-Free Network Analysis](#scale-free-network-analysis)
3. [Graph Generation](#graph-generation)
   - [Creating Graphs from Text](#creating-graphs-from-text)
   - [Adding Subgraphs to Existing Graphs](#adding-subgraphs-to-existing-graphs)
4. [Graph Tools](#graph-tools)
   - [Node Embeddings](#node-embeddings)
   - [Graph Visualization](#graph-visualization)
   - [Graph Statistics, Exporting/Rendering, and Plots](#graph-statistics-and-plots)
   - [Graph Simplification](#graph-simplification)
5. [Conversational Agents](#conversational-agents)
   - [ConversationAgent Class](#conversationagent-class)
   - [Conversation Simulation](#conversation-simulation)
   - [Conversation Summarization](#conversation-summarization)
   - [Question Answering with Agents](#question-answering-with-agents)
6. [Full API](#full_api)

## Introduction <a name="introduction"></a>

This API documentation provides an overview of the functions and classes available in the graph analysis and reasoning code. The code is organized into several files, each focusing on a specific aspect of graph analysis, generation, and conversational agents.

## Graph Analysis <a name="graph-analysis"></a>

The `graph_analysis.py` file contains functions for analyzing graphs, including shortest path finding, path finding with reasoning, community detection, and scale-free network analysis.

### Shortest Path Functions <a name="shortest-path-functions"></a>

- `find_shortest_path`: Finds the shortest path between two nodes in a graph.
- `find_shortest_path_with2hops`: Finds the shortest path considering nodes within 2 hops.
- `find_N_paths`: Finds N shortest paths between two nodes.

### Path Finding and Reasoning <a name="path-finding-and-reasoning"></a>

- `find_path`: Finds a path between two keywords using best fitting nodes.
- `find_path_and_reason`: Finds a path and reasons over it using a language model.
- `find_path_with_relations_and_reason_combined`: Finds paths, reasons over them, considering multiple paths.

### Community Detection and Analysis <a name="community-detection-and-analysis"></a>

- `describe_communities`: Detects and describes the top N communities in the graph.
- `describe_communities_with_plots`: Detects, describes, and plots the top N communities.
- `describe_communities_with_plots_complex`: More detailed community analysis with plots.

### Scale-Free Network Analysis <a name="scale-free-network-analysis"></a>

- `is_scale_free`: Determines if the network is scale-free using the powerlaw package.

## Graph Generation <a name="graph-generation"></a>

The `graph_generation.py` file provides functions for creating graphs from text and adding subgraphs to existing graphs.

### Creating Graphs from Text <a name="creating-graphs-from-text"></a>

- `make_graph_from_text`: Creates a graph from input text.

### Adding Subgraphs to Existing Graphs <a name="adding-subgraphs-to-existing-graphs"></a>

- `add_new_subgraph_from_text`: Adds a new subgraph to an existing graph based on input text.

## Graph Tools <a name="graph-tools"></a>

The `graph_tools.py` file offers various tools for working with graphs, including saving graph in various file formats (including HTML), node embeddings, graph visualization, graph statistics, and graph simplification.

### Node Embeddings <a name="node-embeddings"></a>

- `generate_node_embeddings`: Generates node embeddings using a language model.
- `save_embeddings`: Saves node embeddings to a file.
- `load_embeddings`: Loads node embeddings from a file.
- `find_best_fitting_node`: Finds the best fitting node for a given keyword.
- `find_best_fitting_node_list`: Finds the N best fitting nodes for a given keyword.

### Graph Visualization <a name="graph-visualization"></a>

- `visualize_embeddings_2d`: Visualizes node embeddings in 2D.
- `visualize_embeddings_2d_notext`: Visualizes node embeddings in 2D without text labels.
- `visualize_embeddings_2d_pretty`: Visualizes node embeddings in 2D with a pretty style.
- `visualize_embeddings_2d_pretty_and_sample`: Visualizes node embeddings in 2D with a pretty style and outputs samples for each cluster.

### Graph Statistics, Exporting/Rendering, and Plots <a name="graph-statistics-and-plots"></a>

- `graph_statistics_and_plots_for_large_graphs`: Calculates graph statistics and creates visualizations for large graphs.
- `make_HTML`: Saves graph as HTML file.

### Graph Simplification <a name="graph-simplification"></a>

- `simplify_graph`: Simplifies a graph by merging similar nodes.
- `remove_small_fragents`: Removes small fragments from a graph.
- `update_node_embeddings`: Updates node embeddings for a new graph.

## X-LoRA Tools <a name="x-lora-tools"></a>

The `xlora_tools.py` file contains functions for plotting the scalings of an X-LoRA model.

### Plotting Model Scalings <a name="plotting-model-scalings"></a>

- `plot_scalings`: Plots the scalings of an X-LoRA model in various ways.
- `plot_scalings_from_tensor`: Plots the scalings from a tensor in various ways.

## Conversational Agents <a name="conversational-agents"></a>

The `agents.py` file provides classes and functions for creating and working with conversational agents.

### ConversationAgent Class <a name="conversationagent-class"></a>

- `ConversationAgent`: Represents a conversational agent.

### Conversation Simulation <a name="conversation-simulation"></a>

- `conversation_simulator`: Simulates a conversation between agents.

### Conversation Summarization <a name="conversation-summarization"></a>

- `read_and_summarize`: Reads and summarizes a conversation.

### Question Answering with Agents <a name="question-answering-with-agents"></a>

- `answer_question`: Answers a question using a conversation between two agents.

## Full API Documentation <a name="full_api"></a>

## graph_analysis.py

### `find_shortest_path(G, source='graphene', target='complexity', verbatim=True, data_dir='./')`
- **Description:** Finds the shortest path between two nodes in a graph.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `source` (str): The source node. Default is 'graphene'.
  - `target` (str): The target node. Default is 'complexity'.
  - `verbatim` (bool): Whether to print verbose output. Default is True.
  - `data_dir` (str): The directory to save output files. Default is './'.
- **Returns:** 
  - `path` (list): The shortest path from source to target.
  - `path_graph` (networkx.Graph): The subgraph containing the shortest path.
  - `shortest_path_length` (int): The length of the shortest path.
  - `fname` (str): The filename of the saved HTML file.
  - `graph_GraphML` (str): The filename of the saved GraphML file.

### `find_shortest_path_with2hops(G, source='graphene', target='complexity', second_hop=True, verbatim=True, data_dir='./', save_files=True)`
- **Description:** Finds the shortest path between two nodes considering nodes within 2 hops.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `source` (str): The source node. Default is 'graphene'.
  - `target` (str): The target node. Default is 'complexity'.
  - `second_hop` (bool): Whether to consider nodes within 2 hops. Default is True.
  - `verbatim` (bool): Whether to print verbose output. Default is True.
  - `data_dir` (str): The directory to save output files. Default is './'.
  - `save_files` (bool): Whether to save output files. Default is True.
- **Returns:**
  - `path` (list): The shortest path from source to target.
  - `path_graph` (networkx.Graph): The subgraph containing the shortest path and nodes within 2 hops.
  - `shortest_path_length` (int): The length of the shortest path.
  - `fname` (str): The filename of the saved HTML file. None if `save_files` is False.
  - `graph_GraphML` (str): The filename of the saved GraphML file. None if `save_files` is False.

### `find_N_paths(G, source='graphene', target='complexity', N=5)`
- **Description:** Finds N shortest paths between two nodes.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `source` (str): The source node. Default is 'graphene'.
  - `target` (str): The target node. Default is 'complexity'.
  - `N` (int): The number of shortest paths to find. Default is 5.
- **Returns:**
  - `sampled_paths` (list): A list of the N shortest paths from source to target.
  - `fname_list` (list): A list of filenames of the saved HTML files for each path.

### `find_all_triplets(G)`
- **Description:** Finds all connected triplets of nodes in the graph.
- **Input:**
  - `G` (networkx.Graph): The input graph.
- **Returns:**
  - `triplets` (list): A list of all connected triplets of nodes in the graph.

### `print_node_pairs_edge_title(G)`
- **Description:** Prints node pairs with their edge titles.
- **Input:**
  - `G` (networkx.Graph): The input graph.
- **Returns:**
  - `pairs_and_titles` (list): A list of node pairs with their edge titles.

### `find_path(G, node_embeddings, tokenizer, model, keyword_1='music and sound', keyword_2='graphene', verbatim=True, second_hop=False, data_dir='./', similarity_fit_ID_node_1=0, similarity_fit_ID_node_2=0, save_files=True)`
- **Description:** Finds a path between two keywords using best fitting nodes.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `node_embeddings` (dict): A dictionary of node embeddings.
  - `tokenizer`: The tokenizer for the language model.
  - `model`: The language model.
  - `keyword_1` (str): The first keyword. Default is 'music and sound'.
  - `keyword_2` (str): The second keyword. Default is 'graphene'.
  - `verbatim` (bool): Whether to print verbose output. Default is True.
  - `second_hop` (bool): Whether to consider nodes within 2 hops. Default is False.
  - `data_dir` (str): The directory to save output files. Default is './'.
  - `similarity_fit_ID_node_1` (int): The index of the best fitting node for keyword_1. Default is 0.
  - `similarity_fit_ID_node_2` (int): The index of the best fitting node for keyword_2. Default is 0.
  - `save_files` (bool): Whether to save output files. Default is True.
- **Returns:**
  - `(best_node_1, best_similarity_1, best_node_2, best_similarity_2)` (tuple): The best fitting nodes and their similarities for each keyword.
  - `path` (list): The path from best_node_1 to best_node_2.
  - `path_graph` (networkx.Graph): The subgraph containing the path.
  - `shortest_path_length` (int): The length of the path.
  - `fname` (str): The filename of the saved HTML file. None if `save_files` is False.
  - `graph_GraphML` (str): The filename of the saved GraphML file. None if `save_files` is False.

### `describe_communities(G, N=10)`
- **Description:** Detects and describes the top N communities in the graph.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `N` (int): The number of top communities to describe. Default is 10.
- **Returns:** None. Prints the description of the top N communities.

### `describe_communities_with_plots(G, N=10, N_nodes=5, data_dir='./')`
- **Description:** Detects, describes and plots the top N communities in the graph.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `N` (int): The number of top communities to describe and plot. Default is 10.
  - `N_nodes` (int): The number of top nodes to highlight per community. Default is 5.
  - `data_dir` (str): The directory to save output files. Default is './'.
- **Returns:** None. Saves the community size plot and the combined plot of top nodes by degree for each community.

### `describe_communities_with_plots_complex(G, N=10, N_nodes=5, data_dir='./')`
- **Description:** Performs a more detailed community analysis with plots.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `N` (int): The number of top communities to describe and plot. Default is 10.
  - `N_nodes` (int): The number of top nodes to highlight per community. Default is 5.
  - `data_dir` (str): The directory to save output files. Default is './'.
- **Returns:** None. Saves various plots for community analysis.

### `is_scale_free(G, plot_distribution=True, data_dir='./', manual_xmin=None)`
- **Description:** Determines if the network G is scale-free using the powerlaw package.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `plot_distribution` (bool): Whether to plot the degree distribution with the power-law fit. Default is True.
  - `data_dir` (str): The directory to save output files. Default is './'.
  - `manual_xmin` (int): Manually set the xmin value for power-law fitting.
- **Returns:**
  - `is_scale_free` (bool): Whether the network is scale-free.
  - `fit` (powerlaw.Fit): The powerlaw fit object.

### `print_path_with_edges_as_list(G, path, keywords_separator=' --> ')`
- **Description:** Prints a path with nodes and edge titles as a list.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `path` (list): The path to print.
  - `keywords_separator` (str): The separator for the keywords in the output string. Default is ' --> '.
- **Returns:**
  - `path_elements` (list): The path elements as a list.
  - `as_string` (str): The path elements as a string.

### `find_path_and_reason(G, node_embeddings, tokenizer, model, generate, keyword_1='music and sound', keyword_2='apples', include_keywords_as_nodes=True, inst_prepend='', graph_analysis_type='path and relations', instruction='Now, reason over them and propose a research hypothesis.', verbatim=False, N_limit=None, temperature=0.3, keywords_separator=' --> ', system_prompt='You are a scientist who uses logic and reasoning.', max_tokens=4096, prepend='You are given a set of information from a graph that describes the relationship between materials, structure, properties, and properties. You analyze these logically through reasoning.\n\n', similarity_fit_ID_node_1=0, similarity_fit_ID_node_2=0, save_files=True, data_dir='./', visualize_paths_as_graph=True, display_graph=True, words_per_line=2)`
- **Description:** Finds a path between keywords and reasons over it using LLM.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `node_embeddings` (dict): A dictionary of node embeddings.
  - `tokenizer`: The tokenizer for the language model.
  - `model`: The language model.
  - `generate`: The generate function for the language model.
  - `keyword_1` (str): The first keyword. Default is 'music and sound'.
  - `keyword_2` (str): The second keyword. Default is 'apples'.
  - `include_keywords_as_nodes` (bool): Whether to include the keywords as nodes in the path. Default is True.
  - `inst_prepend` (str): The instruction to prepend to the generate function. Default is ''.
  - `graph_analysis_type` (str): The type of graph analysis to perform. Default is 'path and relations'.
  - `instruction` (str): The instruction for reasoning. Default is 'Now, reason over them and propose a research hypothesis.'.
  - `verbatim` (bool): Whether to print verbose output. Default is False.
  - `N_limit` (int): The maximum number of nodes to include in the path. Default is None (no limit).
  - `temperature` (float): The temperature for the generate function. Default is 0.3.
  - `keywords_separator` (str): The separator for the keywords in the output string. Default is ' --> '.
  - `system_prompt` (str): The system prompt for the generate function. Default is 'You are a scientist who uses logic and reasoning.'.
  - `max_tokens` (int): The maximum number of tokens for the generate function. Default is 4096.
  - `prepend` (str): The string to prepend to the generate function. Default is 'You are given a set of information from a graph that describes the relationship between materials, structure, properties, and properties. You analyze these logically through reasoning.\n\n'.
  - `similarity_fit_ID_node_1` (int): The index of the best fitting node for keyword_1. Default is 0.
  - `similarity_fit_ID_node_2` (int): The index of the best fitting node for keyword_2. Default is 0.
  - `save_files` (bool): Whether to save output files. Default is True.
  - `data_dir` (str): The directory to save output files. Default is './'.
  - `visualize_paths_as_graph` (bool): Whether to visualize the paths as a graph. Default is True.
  - `display_graph` (bool): Whether to display the graph. Default is True.
  - `words_per_line` (int): The number of words per line for the graph visualization. Default is 2.
- **Returns:**
  - `response` (str): The response from the generate function.
  - `(best_node_1, best_similarity_1, best_node_2, best_similarity_2)` (tuple): The best fitting nodes and their similarities for each keyword.
  - `path` (list): The path from best_node_1 to best_node_2.
  - `path_graph` (networkx.Graph): The subgraph containing the path.
  - `shortest_path_length` (int): The length of the path.
  - `fname` (str): The filename of the saved HTML file. None if `save_files` is False.
  - `graph_GraphML` (str): The filename of the saved GraphML file. None if `save_files` is False.

### `find_path_with_relations_and_reason_combined(G, node_embeddings, tokenizer, model, generate, keyword_1='music and sound', keyword_2='apples', include_keywords_as_nodes=True, inst_prepend='', instruction='Now, reason over them and propose a research hypothesis.', verbatim=False, N_limit=None, temperature=0.3, keywords_separator=' --> ', system_prompt='You are a scientist who uses logic and reasoning.', max_tokens=4096, prepend='You are given a set of information from a graph that describes the relationship between materials, structure, properties, and properties. You analyze these logically through reasoning.\n\n', num_paths=2, include_all_possible=False, data_dir='./', save_files=False, visualize_paths_as_graph=False, display_graph=True, words_per_line=2)`
- **Description:** Finds paths between keywords, reasons over them, considering multiple paths.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `node_embeddings` (dict): A dictionary of node embeddings.
  - `tokenizer`: The tokenizer for the language model.
  - `model`: The language model.
  - `generate`: The generate function for the language model.
  - `keyword_1` (str): The first keyword. Default is 'music and sound'.
  - `keyword_2` (str): The second keyword. Default is 'apples'.
  - `include_keywords_as_nodes` (bool): Whether to include the keywords as nodes in the paths. Default is True.
  - `inst_prepend` (str): The instruction to prepend to the generate function. Default is ''.
  - `instruction` (str): The instruction for reasoning. Default is 'Now, reason over them and propose a research hypothesis.'.
  - `verbatim` (bool): Whether to print verbose output. Default is False.
  - `N_limit` (int): The maximum number of nodes to include in each path. Default is None (no limit).
  - `temperature` (float): The temperature for the generate function. Default is 0.3.
  - `keywords_separator` (str): The separator for the keywords in the output string. Default is ' --> '.
  - `system_prompt` (str): The system prompt for the generate function. Default is 'You are a scientist who uses logic and reasoning.'.
  - `max_tokens` (int): The maximum number of tokens for the generate function. Default is 4096.
  - `prepend` (str): The string to prepend to the generate function. Default is 'You are given a set of information from a graph that describes the relationship between materials, structure, properties, and properties. You analyze these logically through reasoning.\n\n'.
  - `num_paths` (int): The number of paths to find. Default is 2.
  - `include_all_possible` (bool): Whether to include all possible combinations of paths. Default is False.
  - `data_dir` (str): The directory to save output files. Default is './'.
  - `save_files` (bool): Whether to save output files. Default is False.
  - `visualize_paths_as_graph` (bool): Whether to visualize the paths as a graph. Default is False.
  - `display_graph` (bool): Whether to display the graph. Default is True.
  - `words_per_line` (int): The number of words per line for the graph visualization. Default is 2.
- **Returns:**
  - `response` (str): The response from the generate function.

## graph_generation.py

### `make_graph_from_text(txt, generate, include_contextual_proximity=False, graph_root='graph_root', chunk_size=2500, chunk_overlap=0, repeat_refine=0, verbatim=False, data_dir='./data_output_KG/', save_PDF=False, save_HTML=True)`
- **Description:** Creates a graph from input text.
- **Input:**
  - `txt` (str): The input text to generate the graph from.
  - `generate`: The generate function for the language model.
  - `include_contextual_proximity` (bool): Whether to include contextual proximity edges. Default is False.
  - `graph_root` (str): The root name for the generated graph files. Default is 'graph_root'.
  - `chunk_size` (int): The size of each chunk of text to process. Default is 2500.
  - `chunk_overlap` (int): The overlap between chunks of text. Default is 0.
  - `repeat_refine` (int): The number of times to repeat the graph refinement process. Default is 0.
  - `verbatim` (bool): Whether to print verbose output. Default is False.
  - `data_dir` (str): The directory to save output files. Default is './data_output_KG/'.
  - `save_PDF` (bool): Whether to save the graph as a PDF file. Default is False.
  - `save_HTML` (bool): Whether to save the graph as an HTML file. Default is True.
- **Returns:**
  - `graph_HTML` (str): The filename of the saved HTML file.
  - `graph_GraphML` (str): The filename of the saved GraphML file.
  - `G` (networkx.Graph): The generated graph.
  - `net` (pyvis.network.Network): The Pyvis network object for the graph.
  - `output_pdf` (str): The filename of the saved PDF file. None if `save_PDF` is False.

### `add_new_subgraph_from_text(txt, generate, node_embeddings, tokenizer, model, original_graph_path_and_fname, data_dir_output='./data_temp/', verbatim=True, size_threshold=10, chunk_size=10000, do_Louvain_on_new_graph=True, include_contextual_proximity=False, repeat_refine=0, similarity_threshold=0.95, simplify_graph=True, return_only_giant_component=False, save_common_graph=True, G_to_add=None, graph_GraphML_to_add=None)`
- **Description:** Adds a new subgraph to an existing graph based on input text.
- **Input:**
  - `txt` (str): The input text to generate the subgraph from.
  - `generate`: The generate function for the language model.
  - `node_embeddings` (dict): A dictionary of node embeddings.
  - `tokenizer`: The tokenizer for the language model.
  - `model`: The language model.
  - `original_graph_path_and_fname` (str): The path and filename of the original graph to add the subgraph to.
  - `data_dir_output` (str): The directory to save output files. Default is './data_temp/'.
  - `verbatim` (bool): Whether to print verbose output. Default is True.
  - `size_threshold` (int): The minimum size of connected components to keep in the graph. Default is 10.
  - `chunk_size` (int): The size of each chunk of text to process. Default is 10000.
  - `do_Louvain_on_new_graph` (bool): Whether to perform Louvain community detection on the new graph. Default is True.
  - `include_contextual_proximity` (bool): Whether to include contextual proximity edges. Default is False.
  - `repeat_refine` (int): The number of times to repeat the graph refinement process. Default is 0.
  - `similarity_threshold` (float): The similarity threshold for simplifying the graph. Default is 0.95.
  - `simplify_graph` (bool): Whether to simplify the graph by merging similar nodes. Default is True.
  - `return_only_giant_component` (bool): Whether to return only the giant component of the graph. Default is False.
  - `save_common_graph` (bool): Whether to save a graph of the common nodes between the original and new graphs. Default is True.
  - `G_to_add` (networkx.Graph): An optional graph to add to the original graph instead of generating a new one. Default is None.
  - `graph_GraphML_to_add` (str): An optional GraphML file to load a graph from instead of generating a new one. Default is None.
- **Returns:**
  - `graph_GraphML` (str): The filename of the saved GraphML file for the combined graph.
  - `G_new` (networkx.Graph): The combined graph.
  - `G_loaded` (networkx.Graph): The loaded graph to add to the original graph.
  - `G` (networkx.Graph): The original graph.
  - `node_embeddings` (dict): The updated node embeddings.
  - `res` (dict): The graph statistics for the combined graph.

## graph_tools.py

### `make_HTML(graph, data_dir, graph_root)`
- **Description:** Saves graph as HTML file for easy visualization in a browser.
- **Input:**
  - `graph` (networkx.Graph): The input graph.
  - `data_dir`: Directory to save HTML graph file in.
  - `graph_rool`: Root for file name.
- **Returns:**
  - `graph_HTML`: File name of graph in HTML format, as `f'{data_dir}/{graph_root}_graphHTML.html`.
    
### `generate_node_embeddings(graph, tokenizer, model)`
- **Description:** Generates node embeddings using a LLM.
- **Input:**
  - `graph` (networkx.Graph): The input graph.
  - `tokenizer`: The tokenizer for the language model.
  - `model`: The language model.
- **Returns:**
  - `embeddings` (dict): A dictionary of node embeddings.

### `save_embeddings(embeddings, file_path)`
- **Description:** Saves node embeddings to a file.
- **Input:**
  - `embeddings` (dict): A dictionary of node embeddings.
  - `file_path` (str): The path to save the embeddings to.
- **Returns:** None.

### `load_embeddings(file_path)`
- **Description:** Loads node embeddings from a file.
- **Input:**
  - `file_path` (str): The path to load the embeddings from.
- **Returns:**
  - `embeddings` (dict): A dictionary of node embeddings.

### `find_best_fitting_node(keyword, embeddings, tokenizer, model)`
- **Description:** Finds the best fitting node for a given keyword.
- **Input:**
  - `keyword` (str): The keyword to find the best fitting node for.
  - `embeddings` (dict): A dictionary of node embeddings.
  - `tokenizer`: The tokenizer for the language model.
  - `model`: The language model.
- **Returns:**
  - `best_node` (str): The best fitting node for the keyword.
  - `best_similarity` (float): The similarity score for the best fitting node.

### `find_best_fitting_node_list(keyword, embeddings, tokenizer, model, N_samples=5)`
- **Description:** Finds the N best fitting nodes for a given keyword.
- **Input:**
  - `keyword` (str): The keyword to find the best fitting nodes for.
  - `embeddings` (dict): A dictionary of node embeddings.
  - `tokenizer`: The tokenizer for the language model.
  - `model`: The language model.
  - `N_samples` (int): The number of best fitting nodes to return. Default is 5.
- **Returns:**
  - `best_nodes` (list): A list of tuples containing the best fitting nodes and their similarity scores.

### `visualize_embeddings_2d(embeddings, data_dir='./')`
- **Description:** Visualizes node embeddings in 2D.
- **Input:**
  - `embeddings` (dict): A dictionary of node embeddings.
  - `data_dir` (str): The directory to save output files. Default is './'.
- **Returns:** None. Saves a 2D visualization of the node embeddings.

### `visualize_embeddings_2d_notext(embeddings, n_clusters=3, data_dir='./')`
- **Description:** Visualizes node embeddings in 2D without text labels.
- **Input:**
  - `embeddings` (dict): A dictionary of node embeddings.
  - `n_clusters` (int): The number of clusters to use for clustering the embeddings. Default is 3.
  - `data_dir` (str): The directory to save output files. Default is './'.
- **Returns:** None. Saves a 2D visualization of the node embeddings without text labels.

### `visualize_embeddings_2d_pretty(embeddings, n_clusters=3, data_dir='./')`
- **Description:** Visualizes node embeddings in 2D with a pretty style.
- **Input:**
  - `embeddings` (dict): A dictionary of node embeddings.
  - `n_clusters` (int): The number of clusters to use for clustering the embeddings. Default is 3.
  - `data_dir` (str): The directory to save output files. Default is './'.
- **Returns:** None. Saves a pretty 2D visualization of the node embeddings.

### `visualize_embeddings_2d_pretty_and_sample(embeddings, n_clusters=3, n_samples=5, data_dir='./', alpha=0.7, edgecolors='none', s=50)`
- **Description:** Visualizes node embeddings in 2D with a pretty style and outputs samples for each cluster.
- **Input:**
  - `embeddings` (dict): A dictionary of node embeddings.
  - `n_clusters` (int): The number of clusters to use for clustering the embeddings. Default is 3.
  - `n_samples` (int): The number of samples to output for each cluster. Default is 5.
  - `data_dir` (str): The directory to save output files. Default is './'.
  - `alpha` (float): The alpha value for the scatter plot. Default is 0.7.
  - `edgecolors` (str): The edge color for the scatter plot. Default is 'none'.
  - `s` (int): The size of the markers in the scatter plot. Default is 50.
- **Returns:** None. Saves a pretty 2D visualization of the node embeddings and outputs samples for each cluster.

### `graph_statistics_and_plots_for_large_graphs(G, data_dir='./', include_centrality=False, make_graph_plot=False, root='graph')`
- **Description:** Calculates graph statistics and creates visualizations for large graphs.
- **Input:**
  - `G` (networkx.Graph): The input graph.
  - `data_dir` (str): The directory to save output files. Default is './'.
  - `include_centrality` (bool): Whether to include centrality measures in the statistics. Default is False.
  - `make_graph_plot` (bool): Whether to create a plot of the graph. Default is False.
  - `root` (str): The root name for the output files. Default is 'graph'.
- **Returns:**
  - `statistics` (dict): A dictionary of graph statistics.
  - `centrality` (dict): A dictionary of centrality measures. None if `include_centrality` is False.

### `simplify_graph(graph_, node_embeddings, tokenizer, model, similarity_threshold=0.9, use_llm=False, data_dir_output='./', graph_root='simple_graph', verbatim=False, max_tokens=2048, temperature=0.3, generate=None)`
- **Description:** Simplifies a graph by merging similar nodes.
- **Input:**
  - `graph_` (networkx.Graph): The input graph.
  - `node_embeddings` (dict): A dictionary of node embeddings.
  - `tokenizer`: The tokenizer for the language model.
  - `model`: The language model.
  - `similarity_threshold` (float): The similarity threshold for merging nodes. Default is 0.9.
  - `use_llm` (bool): Whether to use a language model to rename merged nodes. Default is False.
  - `data_dir_output` (str): The directory to save output files. Default is './'.
  - `graph_root` (str): The root name for the output files. Default is 'simple_graph'.
  - `verbatim` (bool): Whether to print verbose output. Default is False.
  - `max_tokens` (int): The maximum number of tokens to generate for renaming nodes. Default is 2048.
  - `temperature` (float): The temperature for the language model. Default is 0.3.
  - `generate`: The generate function for the language model. Default is None.
- **Returns:**
  - `new_graph` (networkx.Graph): The simplified graph.
  - `updated_embeddings` (dict): The updated node embeddings after simplification.

### `remove_small_fragents(G_new, size_threshold)`
- **Description:** Removes small fragments from a graph.
- **Input:**
  - `G_new` (networkx.Graph): The input graph.
  - `size_threshold` (int): The minimum size of connected components to keep in the graph.
- **Returns:**
  - `G_new` (networkx.Graph): The graph with small fragments removed.

### `update_node_embeddings(embeddings, graph_new, tokenizer, model, remove_embeddings_for_nodes_no_longer_in_graph=True, verbatim=False)`
- **Description:** Updates node embeddings for a new graph.
- **Input:**
  - `embeddings` (dict): A dictionary of node embeddings.
  - `graph_new` (networkx.Graph): The updated graph.
  - `tokenizer`: The tokenizer for the language model.
  - `model`: The language model.
  - `remove_embeddings_for_nodes_no_longer_in_graph` (bool): Whether to remove embeddings for nodes that are no longer in the graph. Default is True.
  - `verbatim` (bool): Whether to print verbose output. Default is False.
- **Returns:**
  - `embeddings_updated` (dict): The updated node embeddings.
    
## agents.py

### `ConversationAgent` class
- **Description:** Represents a conversational agent.
- **Initialization:**
  - `chat_model`: The chat model to use for generating responses.
  - `name` (str): The name of the agent.
  - `instructions` (str): The instructions for the agent.
  - `context_turns` (int): The number of turns of context to use for generating responses. Default is 2.
  - `temperature` (float): The temperature for the language model. Default is 0.1.
- **Methods:**
  - `reply(interlocutor_reply=None)`: Generates a response to the given interlocutor reply.
    - `interlocutor_reply` (str): The reply from the interlocutor. Default is None.
    - Returns: The generated response (str).

### `conversation_simulator(bot0, question_gpt, question_gpt_name='Engineer', question_temperature=0.7, question_asker_instructions='You ALWAYS ask tough questions. ', q='What is bioinspiration?', total_turns=5, data_dir='./')`
- **Description:** Simulates a conversation between agents.
- **Input:**
  - `bot0` (ConversationAgent): The first agent in the conversation.
  - `question_gpt`: The language model to use for generating questions.
  - `question_gpt_name` (str): The name of the question-asking agent. Default is 'Engineer'.
  - `question_temperature` (float): The temperature for the question-asking language model. Default is 0.7.
  - `question_asker_instructions` (str): The instructions for the question-asking agent. Default is 'You ALWAYS ask tough questions. '.
  - `q` (str): The initial question for the conversation. Default is 'What is bioinspiration?'.
  - `total_turns` (int): The total number of turns in the conversation. Default is 5.
  - `data_dir` (str): The directory to save output files. Default is './'.
- **Returns:**
  - `conversation_turns` (list): A list of dictionaries representing each turn in the conversation.

### `read_and_summarize(gpt, txt='This is a conversation.', q='')`
- **Description:** Reads and summarizes a conversation.
- **Input:**
  - `gpt`: The language model to use for summarization.
  - `txt` (str): The conversation text. Default is 'This is a conversation.'.
  - `q` (str): The original question. Default is ''.
- **Returns:**
  - `summary` (str): The summary of the conversation.
  - `bullet` (str): The key points of the conversation as bullet points.
  - `takeaway` (str): The most important takeaway from the conversation.

### `answer_question(gpt_question_asker, gpt, q='I have identified this amino acid sequence: AAAAAIIAAAA. How can I use it?', bot_name_1='Biologist', bot_instructions_1='You are a biologist. You are taking part in a discussion, from a life science perspective.\nKeep your answers brief, but accurate, and creative.\n', bot_name_2='Engineer', bot_instructions_2='You are a critical engineer. You are taking part in a discussion, from the perspective of engineering.\nKeep your answers brief, and always challenge statements in a provokative way. As a creative individual, you inject ideas from other fields. ', question_temperature=0.1, conv_temperature=0.3, total_turns=4, delete_last_question=True, save_PDF=True, PDF_name=None, save_dir='./', txt_file_path=None)`
- **Description:** Answers a question using a conversation between two agents.
- **Input:**
  - `gpt_question_asker`: The language model to use for generating questions.
  - `gpt`: The language model to use for generating responses.
  - `q` (str): The initial question. Default is 'I have identified this amino acid sequence: AAAAAIIAAAA. How can I use it?'.
  - `bot_name_1` (str): The name of the first agent. Default is 'Biologist'.
  - `bot_instructions_1` (str): The instructions for the first agent. Default is 'You are a biologist. You are taking part in a discussion, from a life science perspective.\nKeep your answers brief, but accurate, and creative.\n'.
  - `bot_name_2` (str): The name of the second agent. Default is 'Engineer'.
  - `bot_instructions_2` (str): The instructions for the second agent. Default is 'You are a critical engineer. You are taking part in a discussion, from the perspective of engineering.\nKeep your answers brief, and always challenge statements in a provokative way. As a creative individual, you inject ideas from other fields. '.
  - `question_temperature` (float): The temperature for the question-asking language model. Default is 0.1.
  - `conv_temperature` (float): The temperature for the conversation language model. Default is 0.3.
  - `total_turns` (int): The total number of turns in the conversation. Default is 4.
  - `delete_last_question` (bool): Whether to delete the last question from the conversation. Default is True.
  - `save_PDF` (bool): Whether to save the conversation as a PDF file. Default is True.
  - `PDF_name` (str): The name of the PDF file to save. Default is None.
  - `save_dir` (str): The directory to save output files. Default is './'.
  - `txt_file_path` (str): The path to save the conversation as a text file. Default is None.
- **Returns:**
  - `conversation_turns` (list): A list of dictionaries representing each turn in the conversation.
  - `txt` (str): The conversation text.
  - `summary` (str): The summary of the conversation.
  - `bullet` (str): The key points of the conversation as bullet points.
  - `keytakaway` (str): The most important takeaway from the conversation.
  - `integrated` (str): The integrated conversation text with summary, bullet points, and key takeaway.
  - `save_raw_txt` (str): The raw conversation text without markdown formatting.

 
