import torch
import os
from hubbard_utils import cal_sign, cal_mat_el
save_dir = "./inputdata"
def create_and_save_data(U, t, H_filename):
	
	site_num = 2
	NCn=2
	basis_num = NCn*NCn 

	sing_s1 = [0 for h in range(NCn)]
	sing_s1[0]=1
	sing_s1[1]=0

	sing_s2 = [0 for h in range(NCn)]
	sing_s2[0]=0
	sing_s2[1]=1
	#######################################################################################
	edge_index1=torch.tensor([
	        [0, 1], #source node j
	        [1, 0]  #target node i
	    ], dtype=torch.long)
	edge_index2=torch.tensor([
	        [0, 1], #source node j
	        [1, 0]  #target node i
	    ], dtype=torch.long)
	edge_index3=torch.tensor([
	        [0, 1], #source node j
	        [1, 0]  #target node i
	    ], dtype=torch.long)
	edge_index4=torch.tensor([
	        [0, 1], #source node j
	        [1, 0]  #target node i
	    ], dtype=torch.long)
	num_edges_in_one_sample = edge_index1.shape[1]
	edge_index_list = [edge_index1, edge_index2, edge_index3, edge_index4]
	#######################################################################################
	node_features1=torch.tensor([
	        [0, 1, 1, U, 1/site_num],  # Node 0: [total spin, occ. down, occ. up, onsite interaction, # of doublons]
	        [0, 0, 0, 0, 0/site_num],  # Node 1
	    ])
	node_features2=torch.tensor([
	        [-1/2, 1, 0, 0, 0/site_num],  # Node 0
	        [1/2 , 0, 1, 0, 0/site_num],  # Node 1
	    ])
	node_features3=torch.tensor([
	        [1/2 , 0, 1, 0, 0/site_num],  # Node 0
	        [-1/2, 1, 0, 0, 0/site_num],  # Node 1
	    ])
	node_features4=torch.tensor([
	        [0, 0, 0, 0, 0/site_num],  # Node 0
	        [0, 1, 1, U, 1/site_num],  # Node 1
	    ])
	node_features_list = [node_features1, node_features2, node_features3, node_features4]
	
	#######################################################################################
	edge_attr_list = []
	for i_sample in range(basis_num):
		temp_edge_index   = edge_index_list[i_sample]
		temp_node_features= node_features_list[i_sample]
		tensor_list=[]
		for i_edge in range(num_edges_in_one_sample):
			i_node_anni=temp_edge_index[0][i_edge]
			i_node_crea=temp_edge_index[1][i_edge]
			temp_list=[]
			for i_spin in [1,0]: # 1-up 0-down
				i_spin_anni=i_spin
				sign=cal_sign(i_node_crea, i_node_anni, i_spin_anni, temp_node_features, site_num)[0]
				temp_list.append(-t*sign)
			temp_tensor = torch.tensor(temp_list, dtype=torch.float32)
			tensor_list.append(temp_tensor)
		edge_attr_list.append(torch.stack(tensor_list, dim=0))
	#######################################################################################
	H = torch.zeros((basis_num, basis_num), dtype=torch.float32)
	for i in range(basis_num):
		for j in range(basis_num):
				H[i,j]=cal_mat_el(i,j,node_features_list,site_num,U,t)
    #######################################################################################
	edge_index_all    = torch.stack(edge_index_list   , dim=0)
	edge_attr_all     = torch.stack(edge_attr_list    , dim=0)
	node_features_all = torch.stack(node_features_list, dim=0)
	os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성

	torch.save(edge_index_all,    os.path.join(save_dir, "edge_index_all.pt"))
	torch.save(edge_attr_all,     os.path.join(save_dir, "edge_attr_all.pt"))
	torch.save(node_features_all, os.path.join(save_dir, "node_features_all.pt"))
	torch.save(H, os.path.join(save_dir, H_filename))
# create_and_save_data(1,1,'test')