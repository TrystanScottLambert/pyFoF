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
    return val_x,val_y

def get_edges(results_list: List, n_runs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Works out edges and weights of edges for the resulting groups."""
    print('Generating Edges:')
    print('\t 1 of 2: Calculating Pairs')
    _edges_x, _edges_y=[],[]
    for i in tqdm(range(len(results_list))):
        tupples = get_tuples(results_list[i])
        _edges_x += tupples[0]
        _edges_y += tupples[1]
    _edges_x, _edges_y = np.array(_edges_x),np.array(_edges_y)

    print('\t 2 of 2: Calculating Weights')
    weights,edges_x,edges_y=[],[],[]
    checked = np.zeros(len(_edges_x))
    for i in tqdm(range(len(_edges_y))):
        if checked[i]==0:
            x_val = np.where(_edges_x==_edges_x[i])
            y_val = np.where(_edges_y==_edges_y[i])
            val=np.intersect1d(x_val, y_val)
            checked[val] = 1
            number_of_instances = float(len(val))
            percentage_of_instances = number_of_instances/n_runs
            weights.append(percentage_of_instances)
            edges_x.append(edges_x[i])
            edges_y.append(edges_y[i])
    return edges_x, edges_y, weights

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
    print('Identifying Subgraphs:')
    sub_graphs = list(graph.subgraph(c).copy() for c in nx.connected_components(graph))
    return sub_graphs

def find_proper_groups(sub_graph_list: List[nx.Graph]) -> List[nx.Graph]:
    """Remove any pairings have less than 3 members. (i.e. not groups.)"""
    print('Applying > 2 condition for groups.')
    return [sub_graph for sub_graph in sub_graph_list if len(sub_graph.nodes) > 2]

def get_node_arrays(stable_list):
    """makes a list of galaxy ids for every true group."""
    return [list(stable_group.nodes) for stable_group in stable_list]

def get_edges_arrays(stable_list: List[nx.Graph]) -> np.ndarray[List]:
    """Writing the edges as arrays"""
    print('Getting Edge Data')
    edges_array=[]
    for i in tqdm(range(len(stable_list))):
        local_edges_full = nx.get_edge_attributes(stable_list[i],'weight')
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
    return np.unqique(np.sort(get_edges_arrays([graph])))

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

def subgroup_list(graph_list: List[nx.Graph]) -> List[List[nx.Graph]]:
    """Works out the sub groups for a list of graphs."""
    return [sub_groups(graph) for graph in graph_list]


def stabalize(results_list, cutoff, n_runs):
    """ averages over multiple runs, returning final groups."""
    main_graph = generate_main_graph(results_list, n_runs)
    sub_graphs = get_subgraphs(main_graph)
    edge_data = get_edges_arrays(sub_graphs)

    cut_main_graph = cut_edges(main_graph, cutoff)
    sub_graphs = get_subgraphs(cut_main_graph)
    stables = find_proper_groups(sub_graphs)
    stable_arrays = get_node_arrays(stables)

    weights, weights_normed=wc_list(stables)
    sub_groupings = subgroup_list(stables)

    return stable_arrays, edge_data, weights, weights_normed, sub_groupings
