import torch, math, os
import module
from hubbard_utils import cal_sign, cal_mat_el

def create_and_save_data(N, U, t, sample_count):
    if not isinstance(N, int):
        raise TypeError(f"N must be an integer, but got {type(N).__name__}")
    ##########################################################################################
    NCn=math.comb(N, N // 2) # of possible up(down) spin configurations
    basis_num = NCn*NCn   # of basis
    ##########################################################################################
    from itertools import combinations
    num_ones = N // 2
    num_zeros = N - num_ones
    sing = [[0]*N for _ in range(NCn)]
    for idx, ones_positions in enumerate(combinations(range(N), num_ones)):
        row = [0] * N
        for pos in ones_positions:
            row[pos] = 1
        sing[idx] = row
    sing.sort(key=lambda bits: int("".join(map(str,bits)), 2))
    ##########################################################################################
    node_features_list = []
    for idown in range(NCn):
        for iup in range(NCn):
            one_sample_features = []
            for node_idx in range(N):
                occ_down = sing[idown][N-node_idx-1]
                occ_up = sing[iup][N-node_idx-1]
                total_spin = occ_up * 0.5 + occ_down * (-0.5)
                onsite_interaction = math.floor((occ_up + occ_down) / 1.5) * U
                normalized_doublon = math.floor((occ_up + occ_down) / 1.5) / N
    
                one_node_feature = [
                    total_spin,
                    occ_down,
                    occ_up,
                    onsite_interaction,
                    normalized_doublon
                ]
                one_sample_features.append(one_node_feature)
    
            node_tensor = torch.tensor(one_sample_features, dtype=torch.float32)
            node_features_list.append(node_tensor)
    ##########################################################################################
    edge_index_list=[]
    for i_sample in range(basis_num):
        temp_tensor = torch.zeros((2, 2 * N), dtype=torch.long)
        for i_site in range(N):
            temp_tensor[0,i_site  ] =  i_site
            temp_tensor[1,i_site  ] = (i_site+1) % N
            temp_tensor[0,i_site+1] = (i_site+1) % N
            temp_tensor[1,i_site+1] =  i_site
        edge_index_list.append(temp_tensor)
    num_edges_in_one_sample = edge_index_list[1].shape[1]
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
                sign=cal_sign(i_node_crea, i_node_anni, i_spin_anni, temp_node_features, N)[0]
                temp_list.append(-t*sign)
            temp_tensor = torch.tensor(temp_list, dtype=torch.float32)
            tensor_list.append(temp_tensor)
        edge_attr_list.append(torch.stack(tensor_list, dim=0))
    ##########################################################################################
    H = torch.zeros((basis_num, basis_num), dtype=torch.float32)
    for i in range(basis_num):
        for j in range(basis_num):
             H[i,j]=cal_mat_el(i,j,node_features_list,N,U,t)
    ########################################################################################## 
    edge_index_all    = torch.stack(edge_index_list   , dim=0)
    edge_attr_all     = torch.stack(edge_attr_list    , dim=0)
    node_features_all = torch.stack(node_features_list, dim=0)

    os.makedirs(module.input_dir, exist_ok=True)  # 폴더 없으면 생성
    torch.save(edge_index_all   , os.path.join(module.input_dir,    f"edge_index_all_{sample_count}.pt"))
    torch.save(edge_attr_all    , os.path.join(module.input_dir,     f"edge_attr_all_{sample_count}.pt"))
    torch.save(node_features_all, os.path.join(module.input_dir, f"node_features_all_{sample_count}.pt"))
    torch.save(H                , os.path.join(module.input_dir,                 f"H_{sample_count}.pt"))
