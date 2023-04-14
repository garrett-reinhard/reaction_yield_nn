import os, sys
import numpy as np
import torch
from dgl.convert import graph


class GraphDataset():

    def __init__(self, data_id, split_id):

        self._data_id = data_id
        self._split_id = split_id
        self.load()


    def load(self):
        """load Herein, the data is loaded from the npz files that were generated previously
        The npz files take the following format:
        List of Dictionaries:
        rmol_dict: Dictionary of Reactants
            Contains the following information for each reactant
            dict_keys(['n_node', 'n_edge', 'node_attr', 'edge_attr', 'src', 'dst'])
            Previously converted to graph objects
        pmol_dict: Dictionary of Products
            Contains the following information for each product
            dict_keys(['n_node', 'n_edge', 'node_attr', 'edge_attr', 'src', 'dst'])
        reaction_dict: Dictionary of Reaction Smiles
            Contains the following information for the reaction
            dict_keys(['yld', 'rsmi'])

        _extended_summary_
        """
        # Loads the data based on their dataset id
        if self._data_id in [1, 2]:
            [rmol_dict, pmol_dict, reaction_dict] = np.load('./data/dataset_%d_%d.npz' %(self._data_id, self._split_id), allow_pickle=True)['data']
        elif self._data_id == 3:
            [rmol_dict, pmol_dict, reaction_dict] = np.load('./data/test_%d.npz' %self._split_id, allow_pickle=True)['data']
        # Sets variable to integer of max key:value pairs in dictionaries
        self.rmol_max_cnt = len(rmol_dict)
        self.pmol_max_cnt = len(pmol_dict)

        # These extract information previously generated in the graph lists/dictionaries
        # up to the max length of the dictionaries
        # src (source node) and dst (destination)
        self.rmol_n_node = [rmol_dict[j]['n_node'] for j in range(self.rmol_max_cnt)]
        self.rmol_n_edge = [rmol_dict[j]['n_edge'] for j in range(self.rmol_max_cnt)]
        self.rmol_node_attr = [rmol_dict[j]['node_attr'] for j in range(self.rmol_max_cnt)]
        self.rmol_edge_attr = [rmol_dict[j]['edge_attr'] for j in range(self.rmol_max_cnt)]
        self.rmol_src = [rmol_dict[j]['src'] for j in range(self.rmol_max_cnt)]
        self.rmol_dst = [rmol_dict[j]['dst'] for j in range(self.rmol_max_cnt)]
        
        # Process repeated here for the product dictionary
        self.pmol_n_node = [pmol_dict[j]['n_node'] for j in range(self.pmol_max_cnt)]
        self.pmol_n_edge = [pmol_dict[j]['n_edge'] for j in range(self.pmol_max_cnt)]
        self.pmol_node_attr = [pmol_dict[j]['node_attr'] for j in range(self.pmol_max_cnt)]
        self.pmol_edge_attr = [pmol_dict[j]['edge_attr'] for j in range(self.pmol_max_cnt)]
        self.pmol_src = [pmol_dict[j]['src'] for j in range(self.pmol_max_cnt)]
        self.pmol_dst = [pmol_dict[j]['dst'] for j in range(self.pmol_max_cnt)]
        
        # Grabss the values from yield key and reaction smiles key
        self.yld = reaction_dict['yld']
        self.rsmi = reaction_dict['rsmi']

        # Cumulative sum of nodes and edges of the graph objects of reactants?
        # Ask Dave: is this to build "dependencies" and relationships between graphs?
        self.rmol_n_csum = [np.concatenate([[0], np.cumsum(self.rmol_n_node[j])]) for j in range(self.rmol_max_cnt)]
        self.rmol_e_csum = [np.concatenate([[0], np.cumsum(self.rmol_n_edge[j])]) for j in range(self.rmol_max_cnt)]

        self.pmol_n_csum = [np.concatenate([[0], np.cumsum(self.pmol_n_node[j])]) for j in range(self.pmol_max_cnt)]
        self.pmol_e_csum = [np.concatenate([[0], np.cumsum(self.pmol_n_edge[j])]) for j in range(self.pmol_max_cnt)]
        

    def __getitem__(self, idx):
        # Creating a list of graph objects based on the previously established lists
        #  g1 only collects reactants 
        g1 = [graph((self.rmol_src[j][self.rmol_e_csum[j][idx]:self.rmol_e_csum[j][idx+1]],
                     self.rmol_dst[j][self.rmol_e_csum[j][idx]:self.rmol_e_csum[j][idx+1]]
                     ), num_nodes = self.rmol_n_node[j][idx])
              for j in range(self.rmol_max_cnt)]

        # Sets the attributes for the graphs in the reactant graph list      
        for j in range(self.rmol_max_cnt):
            g1[j].ndata['attr'] = torch.from_numpy(self.rmol_node_attr[j][self.rmol_n_csum[j][idx]:self.rmol_n_csum[j][idx+1]]).float()
            g1[j].edata['edge_attr'] = torch.from_numpy(self.rmol_edge_attr[j][self.rmol_e_csum[j][idx]:self.rmol_e_csum[j][idx+1]]).float()
        
        # Creates a list of graph objects for products on previously established lists
        g2 = [graph((self.pmol_src[j][self.pmol_e_csum[j][idx]:self.pmol_e_csum[j][idx+1]],
                     self.pmol_dst[j][self.pmol_e_csum[j][idx]:self.pmol_e_csum[j][idx+1]]
                     ), num_nodes = self.pmol_n_node[j][idx])
              for j in range(self.pmol_max_cnt)]
        
        # Sets the attributes for the graphs in the product graph list
        for j in range(self.pmol_max_cnt):
            g2[j].ndata['attr'] = torch.from_numpy(self.pmol_node_attr[j][self.pmol_n_csum[j][idx]:self.pmol_n_csum[j][idx+1]]).float()
            g2[j].edata['edge_attr'] = torch.from_numpy(self.pmol_edge_attr[j][self.pmol_e_csum[j][idx]:self.pmol_e_csum[j][idx+1]]).float()

        label = self.yld[idx]
        
        return *g1, *g2, label
        
        
    def __len__(self):

        return self.yld.shape[0]
