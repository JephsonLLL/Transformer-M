# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from functools import lru_cache
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos

import os
import shutil
import os.path as osp
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from multiprocessing import Pool

from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
from rdkit import Chem

import tarfile

import pdb

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype = np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)

        return graph
    except:
        return None

def mol2graph(mol):
    try:

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype = np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

        # positions
        positions = mol.GetConformer().GetPositions()

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)
        graph['position'] = positions

        return graph
    except:
        return None

class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(smiles2graph, smiles_list)

            for i, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                try:
                    data = Data()

                    homolumogap = homolumogap_list[i]

                    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])

                    data_list.append(data)
                except:
                    continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict


class PygPCQM4Mv2PosDataset(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = root
        #self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
        self.pos_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'

        # check version and update if necessary
        '''
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)
        '''
        super(PygPCQM4Mv2PosDataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'train_test_344.csv'
        #return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

        if decide_download(self.pos_url):
            path = download_url(self.pos_url, self.original_root)
            tar = tarfile.open(path, 'r:gz')
            filenames = tar.getnames()
            for file in filenames:
                tar.extract(file, self.original_root)
            tar.close()
            os.unlink(path)
        else:
            print('Stop download')
            exit(-1)

            

    def mean(self, target, idxs):
        y = torch.cat([self.get(i).y for i in idxs], dim=0)
        return y.mean().item()

    def std(self, target, idxs):
        y = torch.cat([self.get(i).y for i in idxs], dim=0)
        return y.std().item()



    def process(self):
        
        data_df = pd.read_csv(osp.join(self.raw_dir, 'train_test_344.csv'))
        #data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        Donor_graph_pos_list = Chem.SDMolSupplier(osp.join(self.original_root, 'opv.sdf'))
        #graph_pos_list = Chem.SDMolSupplier(osp.join(self.original_root, 'pcqm4m-v2-train.sdf'))

        PC61BM = "COC(=O)CCCC1(C23C14C5=C6C7=C8C5=C9C1=C5C%10=C%11C%12=C%13C%10=C%10C1=C8C1=C%10C8=C%10C%14=C%15C%16=C%17C(=C%12C%12=C%17C%17=C%18C%16=C%16C%15=C%15C%10=C1C7=C%15C1=C%16C(=C%18C7=C2C2=C%10C(=C5C9=C42)C%11=C%12C%10=C%177)C3=C16)C%14=C%138)C1=CC=CC=C1"
        PC71BM = "COC(=O)CCCC1(C23C14C5=C6C7=C8C9=C1C%10=C%11C9=C9C%12=C%13C%14=C%15C%16=C%17C%18=C%19C%20=C%21C%22=C%23C%24=C%25C%26=C%27C%28=C(C%14=C%14C%12=C%11C%11=C%14C%28=C%26C%12=C%11C%10=C%10C%12=C%25C%23=C%11C%10=C1C7=C%11C%22=C6C4=C%21C%19=C2C%17=C1C%15=C%13C2=C9C8=C5C2=C31)C%16=C%18C%27=C%24%20)C1=CC=CC=C1"
        TiO2 = "O=[Ti]=O"
        C60 = "C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12C5=C%11C4=C3C3=C5C(=C81)C%10=C23"
        PDI = "C1=CC2=C3C(=CC=C4C3=C1C5=C6C4=CC=C7C6=C(C=C5)C(=O)OC7=O)C(=O)OC2=O"
        ICBA = "C1C2C3=CC=CC=C3C1C45C26C7=C8C9=C1C2=C3C8=C8C6=C6C4=C4C%10=C%11C%12=C%13C%14=C%15C%16=C%17C(=C1C1=C2C2=C%18C%19=C%20C2=C3C8=C2C%20=C(C%10=C26)C2=C%11C3=C%13C%15=C6C%17=C1C%181C6(C3=C2%19)C2CC1C1=CC=CC=C21)C1=C%16C2=C%14C(=C%124)C5=C2C7=C19"

        acceptor_dict = {'PC61BM': PC61BM, 'PC71BM':PC71BM, 'TiO2':TiO2, 'C60':C60, 'PDI':PDI, 'ICBA':ICBA}
        acceptor_list = data_df['Acceptors']
        graph_pos_list = []

        for i in range(344):
            acc_mol = Chem.MolFromSmiles(acceptor_dict[acceptor_list[i]])
            combined_mol = Chem.CombineMols(Donor_graph_pos_list[i], acc_mol)
            graph_pos_list.append(combined_mol)
            

        smiles_list = data_df['Donors']
        #PCE
        homolumogap_list = data_df['PCE']
        #smiles_list = data_df['smiles']
        #homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(smiles2graph, smiles_list)

            for i, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                try:
                    data = Data()

                    homolumogap = homolumogap_list[i]

                    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.zeros(data.__num_nodes__, 3).to(torch.float32)

                    data_list.append(data)
                except:
                    continue
   
        print('Extracting 3D positions from SDF files for Training Data...')
        train_data_with_position_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(mol2graph, graph_pos_list)

            for i, graph in tqdm(enumerate(iter), total=len(graph_pos_list)):
                try:
                    data = Data()
                    homolumogap = homolumogap_list[i]

                    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.from_numpy(graph['position']).to(torch.float32)
                    #pdb.set_trace()

                    train_data_with_position_list.append(data)
                except:
                    continue
        data_list = train_data_with_position_list + data_list[len(train_data_with_position_list):]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')

        #pdb.set_trace()
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict


@torch.jit.script
def convert_to_single_emb(x, offset :int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index.to(torch.int64), item.x
    N = x.size(0)
    
    x = convert_to_single_emb(x)
    #pdb.set_trace()
    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())

    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    return item

class MyPygPCQM4MDataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)

class MyPygPCQM4MPosDataset(PygPCQM4Mv2PosDataset):

        

    def download(self):
        super(MyPygPCQM4MPosDataset, self).download()

    def process(self):
        super(MyPygPCQM4MPosDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        '''
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)
        '''
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx

            #pdb.set_trace()
            #self.mean = self.PygPCQM4Mv2PosDataset.mean
            #self.std = self.PygPCQM4Mv2PosDataset.std

            #item.y = item.y[0].unsqueeze(0)
            item.train_mean = self.mean(0, torch.arange(276))
            item.train_std = self.std(0, torch.arange(276))
            item.type = None
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")


if __name__ == "__main__":
    dataset = PygPCQM4Mv2PosDataset()
    print(len(dataset))