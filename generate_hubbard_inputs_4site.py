import torch
import math
import os
from hubbard_utils import cal_sign, cal_mat_el
save_dir = "./inputdata"
def create_and_save_data(U, t, H_filename):
    site_num=4
    NCn=6 # of possible up(down) spin configurations
    basis_num = NCn*NCn # of basis

    sing_s1 = [0 for h in range(NCn)] # single_spin & site 1 & index=basis
    sing_s1[0]=1
    sing_s1[1]=1
    sing_s1[2]=0
    sing_s1[3]=1
    sing_s1[4]=0
    sing_s1[5]=0

    sing_s2 = [0 for h in range(NCn)]
    sing_s2[0]=1
    sing_s2[1]=0
    sing_s2[2]=1
    sing_s2[3]=0
    sing_s2[4]=1
    sing_s2[5]=0

    sing_s3 = [0 for h in range(NCn)]
    sing_s3[0]=0
    sing_s3[1]=1
    sing_s3[2]=1
    sing_s3[3]=0
    sing_s3[4]=0
    sing_s3[5]=1

    sing_s4 = [0 for h in range(NCn)]
    sing_s4[0]=0
    sing_s4[1]=0
    sing_s4[2]=0
    sing_s4[3]=1
    sing_s4[4]=1
    sing_s4[5]=1

    ##########################################################################################
    edge_index_list=[]
    for re1 in range(basis_num):
      edge_index_list.append(torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0], 
            [1, 0, 2, 1, 3, 2, 0, 3]  
        ], dtype=torch.long))
    num_edges_in_one_sample = edge_index_list[1].shape[1]
    ##########################################################################################
    node_features_list=[]  
    for re2 in range(NCn):   
      for re3 in range(NCn):
        node_features_list.append(torch.tensor([
          # [               total spin               , occ. down   , occ. up     ,                onsite interaction              ]
            [sing_s1[re3]*(0.5) + sing_s1[re2]*(-0.5), sing_s1[re2], sing_s1[re3], math.floor((sing_s1[re3] + sing_s1[re2])/1.5)*U, math.floor((sing_s1[re3] + sing_s1[re2])/1.5)/site_num],  # Node 0
            [sing_s2[re3]*(0.5) + sing_s2[re2]*(-0.5), sing_s2[re2], sing_s2[re3], math.floor((sing_s2[re3] + sing_s2[re2])/1.5)*U, math.floor((sing_s2[re3] + sing_s2[re2])/1.5)/site_num],  # Node 1
            [sing_s3[re3]*(0.5) + sing_s3[re2]*(-0.5), sing_s3[re2], sing_s3[re3], math.floor((sing_s3[re3] + sing_s3[re2])/1.5)*U, math.floor((sing_s3[re3] + sing_s3[re2])/1.5)/site_num],  # Node 2
            [sing_s4[re3]*(0.5) + sing_s4[re2]*(-0.5), sing_s4[re2], sing_s4[re3], math.floor((sing_s4[re3] + sing_s4[re2])/1.5)*U, math.floor((sing_s4[re3] + sing_s4[re2])/1.5)/site_num]   # Node 3
        ], dtype=torch.float32))
    ##########################################################################################
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
    ##########################################################################################
    H = torch.zeros((basis_num, basis_num), dtype=torch.float32)
    for i in range(basis_num):
      for j in range(basis_num):
        H[i,j]=cal_mat_el(i,j,node_features_list,site_num,U,t)
    ########################################################################################## 
    edge_index_all    = torch.stack(edge_index_list   , dim=0)
    edge_attr_all     = torch.stack(edge_attr_list    , dim=0)
    node_features_all = torch.stack(node_features_list, dim=0)

    os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성

    torch.save(edge_index_all,    os.path.join(save_dir, "edge_index_all.pt"))
    torch.save(edge_attr_all,     os.path.join(save_dir, "edge_attr_all.pt"))
    torch.save(node_features_all, os.path.join(save_dir, "node_features_all.pt"))
    torch.save(H, os.path.join(save_dir, H_filename))

# create_and_save_data(1,1,'test')