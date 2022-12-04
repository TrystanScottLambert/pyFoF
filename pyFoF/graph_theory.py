"""Group theory module for post-processing all the run data."""

from typing import Tuple, List
import numpy as np
import networkx as nx
from tqdm import tqdm

def get_tuples(array_group: np.ndarray) -> Tuple[List, List]:
    """All possible connections within the local group we are checking."""
    group_of_interest=np.sort(array_group) #ordering is very important for this method
    val_x,val_y=[],[]
    for i in range(len(group_of_interest)-1):
        for j in range(len(group_of_interest)-1-i):
            val_x.append(group_of_interest[i])
            val_y.append(group_of_interest[i+j+1])
    return val_x, val_y

def get_edges(results_list: List, n_runs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Works out edges and weights of edges for the resulting groups."""
    print('Generating Edges:')
    print('\t 1 of 2: Calculating Pairs')
    tuples = [get_tuples(result_list) for result_list in results_list]
    tuple_left = np.concatenate([tup[0] for tup in tuples]).astype(int)
    tuple_right = np.concatenate([tup[1] for tup in tuples]).astype(int)
    tuple_strings = [f'{tuple_left[i]}-{tuple_right[i]}' for i in range(len(tuple_left))]
    print('\t 2 of 2: Calculating Edges')
    number_of_istances = np.unique(tuple_strings, return_counts = True)
    tuples = [tup.split('-') for tup in number_of_istances[0]]
    edges_x = np.array([int(tup[0]) for tup in tuples])
    edges_y = np.array([int(tup[1]) for tup in tuples])
    return edges_x, edges_y, number_of_istances[1] / n_runs

def get_nodes(results_list: List):
    """Finding all the nodes (i.e. galaxies in groups)."""
    print('Generating Nodes:')
    nodes=np.unique(np.concatenate(results_list))
    return nodes

def generate_main_graph(results_list: List, n_runs: int) -> nx.Graph:
    """Makes a graph of all the entire groups results. Lets us use the nx.Graph functions"""
    nodes=get_nodes(results_list)
    edges_x, edges_y, edges_weight=get_edges(results_list, n_runs)
    print('Generating Main Graph:')
    main_graph=nx.Graph()
    print('\t 1 of 2: Implementing nodes')
    main_graph.add_nodes_from(nodes)
    print('\t 2 of 2: Implementing edges')
    for i in tqdm(range(len(edges_x))):
        main_graph.add_edge(edges_x[i],edges_y[i],weight=edges_weight[i])
    return main_graph

def get_subgraphs(graph: nx.Graph):
    """finds all the subgraphs in a graph object."""
    sub_graphs = list(graph.subgraph(c).copy() for c in nx.connected_components(graph))
    return sub_graphs

def find_proper_groups(sub_graph_list: List[nx.Graph]) -> List[nx.Graph]:
    """Remove any pairings have less than 3 members. (i.e. not groups.)"""
    return [sub_graph for sub_graph in sub_graph_list if len(sub_graph.nodes) > 2]

def get_node_arrays(stable_list):
    """makes a list of galaxy ids for every true group."""
    return [list(stable_group.nodes) for stable_group in stable_list]

def get_edges_arrays(stable_list: List[nx.Graph]) -> np.ndarray:
    """Writing the edges as arrays"""
    edges_array=[]
    for stable_graph in stable_list:
        local_edges_full = nx.get_edge_attributes(stable_graph,'weight')
        local_edges_only= list(local_edges_full)
        for edge in local_edges_only:
            local_edge_array=[int(edge[0]), int(edge[1]), float(local_edges_full[edge])]
            edges_array.append(local_edge_array)
    return np.array(edges_array)

def cut_edges(graph, threshold):
    """Removes edges based on the threshold."""
    list_graph=[graph]
    edge_array=get_edges_arrays(list_graph)
    local_nodes=list(graph.nodes())
    new_graph=nx.Graph()
    new_graph.add_nodes_from(local_nodes)
    for edge in edge_array:
        if edge[-1] >= threshold:
            new_graph.add_edge(int(edge[0]), int(edge[1]), weight=edge[2])
    return new_graph

def weighted_centrality(graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """Work out both the weighted centrality and the normalized weighted centrality"""
    graph_nodes=list(graph.nodes())
    sum_weight=np.array(graph.degree(graph_nodes,'weight'))
    sum_weight=sum_weight[:,1]  # strip off the nodes and keep the weightings
    sum_weight_norm=sum_weight.astype(float) / (len(graph_nodes)-1)
    return sum_weight, sum_weight_norm

def wc_list(graph_list: List[nx.Graph]) -> Tuple[np.ndarray, np.ndarray]:
    """return weighted centrality as a list."""
    centralities,normed_centralities = [], []
    for graph in graph_list:
        val = weighted_centrality(graph)
        centralities.append(val[0])
        normed_centralities.append(val[1])
    return centralities, normed_centralities

def ranking_edges(graph):
    """Getting ranking of edges?"""
    return np.unique(np.sort(get_edges_arrays([graph])))

def sub_groups(graph: nx.Graph) -> List[nx.Graph]:
    """I don't know what this is doing."""
    edges_rank = ranking_edges(graph)
    number_subgraphs = []
    subies = []
    for edge in edges_rank:
        local_graph = cut_edges(graph, edge)
        local_graph = get_subgraphs(local_graph)
        local_graph = find_proper_groups(local_graph)
        number_subgraphs.append(len(local_graph))
        subies.append(local_graph)
    number_subgraphs = np.array(number_subgraphs)
    val_subs = np.where(number_subgraphs==np.max(number_subgraphs))[0][0]
    subies = subies[val_subs]
    subies = get_node_arrays(list(subies))
    return subies

def stabalize(results_list, cutoff, n_runs):
    """ averages over multiple runs, returning final groups."""

    main_graph = generate_main_graph(results_list, n_runs)
    sub_graphs = get_subgraphs(main_graph)
    edge_data = get_edges_arrays(sub_graphs)
    cut_main_graph = cut_edges(main_graph, cutoff)
    sub_graphs = get_subgraphs(cut_main_graph)
    stables = find_proper_groups(sub_graphs)
    stable_arrays = get_node_arrays(stables)
    weights, weights_normed = wc_list(stables)

    return stable_arrays, edge_data, weights, weights_normed
