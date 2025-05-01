import torch
def cal_sign(i_node_crea, i_node_anni, i_spin_anni, node_features, site_num):
	# This function determines the sign that multiplies (-t) when -t * c†_(j,σ) * c_(i,σ) is applied to the i-th node.
	# Here, i and j correspond to i_node_anni and i_node_crea  respectively, and i_spin_anni corresponds to i_spin_anni.
	# i_node: 0~(N_site-1)
	# i_spin: 0 (down) or 1 (up)

	# spin conservation
	i_spin_crea=i_spin_anni
	temp_node_features=node_features.clone()
	is_occupied=temp_node_features[i_node_anni,i_spin_anni+1] # which returns 1 when a state (site,spin) is occupied
	 

	if is_occupied!=1:
		sign=0
	else:
		num_occ1=0
		for i_node in range(i_node_anni+1,site_num):
			for i_spin in [0,1]:
				num_occ1=num_occ1+temp_node_features[i_node,i_spin+1]
		if i_spin_anni==0: # special consideration for the same site with (i_node_anni)-th node
			num_occ1=num_occ1+temp_node_features[i_node_anni,1+1]
		temp_node_features[i_node_anni,i_spin_anni+1]=0 # This results from c_(i,σ)c†_(i,σ)=1-c†_(i,σ)c_(i,σ)
			
		is_occupied=temp_node_features[i_node_crea,i_spin_crea+1] # which returns 1 when a state (nearest site,same spin) is occupied  
		if is_occupied==1:
			sign=0
		else:
			num_occ2=0
			for i_node in range(i_node_crea+1,site_num):
				for i_spin in [0,1]:
					num_occ2=num_occ2+temp_node_features[i_node,i_spin+1]
			if i_spin_crea==0:
				num_occ2=num_occ2+temp_node_features[i_node_crea,1+1]
			sign=(-1)**(num_occ1+num_occ2)
			# sign=(num_occ2)
	return sign

def create_and_save_data(U, t):
	
	site_num = 2 # 몇사이트 인가?
	NCn=2
	basis_num = NCn*NCn 

	sing_s1 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
	sing_s1[0]=1
	sing_s1[1]=0

	sing_s2 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
	sing_s2[0]=0
	sing_s2[1]=1


	# 여기까지 각 사이트의 각 스핀의 전자수를 함 써봣다.

	H_list = [[0 for w in range(basis_num)] for h in range(basis_num)] #0으로 다 초기화 합시다.

	H_list[0][0]=U
	H_list[3][3]=U

	# 여기까지 U에 관한 해밀토니안 다 만들었다.

	# spin up 에 대한 hopping 해밀토니안 만들기.

	H_list[0][1]=-t
	H_list[1][0]=-t
	H_list[1][3]=-t
	H_list[3][1]=-t

	# spin down 에 대한 hopping 해밀토니안 만들기.

	H_list[0][2]=t
	H_list[2][0]=t
	H_list[2][3]=t
	H_list[3][2]=t

	H = torch.tensor(H_list) #텐서로 만들기.
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
		    #before changing site numbering)
	        #[0   , 0, 0],  # Node 0: [total spin, number of occupation, onsite interaction]
	        #[0   , 2, U],  # Node 1
	        #-----------------------
	        #before separating spin occupations)
	        #[0   , 2, U],  # Node 0: [total spin, number of occupation, onsite interaction]
	        #[0   , 0, 0],  # Node 1
	        #-----------------------
	        #before adding # of doublons)
	        # [0, 1, 1, U],  # Node 0: [total spin, occ. down, occ. up, onsite interaction]
	        # [0, 0, 0, 0],  # Node 1
	        #-----------------------
	        [0, 1, 1, U, 1/site_num],  # Node 0: [total spin, occ. down, occ. up, onsite interaction, # of doubleron]
	        [0, 0, 0, 0, 0/site_num],  # Node 1
	    ])
	node_features2=torch.tensor([
	        [1/2 , 0, 1, 0, 0/site_num],  # Node 0: [total spin, occ. down, occ. up, onsite interaction, # of doubleron]]
	        [-1/2, 1, 0, 0, 0/site_num],  # Node 1
	    ])
	node_features3=torch.tensor([
	        [-1/2, 1, 0, 0, 0/site_num],  # Node 0: [total spin, occ. down, occ. up, onsite interaction, # of doubleron]]
	        [1/2 , 0, 1, 0, 0/site_num],  # Node 1
	    ])
	node_features4=torch.tensor([
	        [0, 0, 0, 0, 0/site_num],  # Node 0: [total spin, occ. down, occ. up, onsite interaction, # of doubleron]]
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
				sign=cal_sign(i_node_crea, i_node_anni, i_spin_anni, temp_node_features, site_num)
				temp_list.append(-t*sign)
			temp_tensor = torch.tensor(temp_list, dtype=torch.float32)
			tensor_list.append(temp_tensor)
		edge_attr_list.append(torch.stack(tensor_list, dim=0))
	#######################################################################################
	edge_index_all    = torch.stack(edge_index_list   , dim=0)
	edge_attr_all     = torch.stack(edge_attr_list    , dim=0)
	node_features_all = torch.stack(node_features_list, dim=0)

	torch.save(edge_index_all   , "edge_index_all.pt   ")
	torch.save(edge_attr_all    , "edge_attr_all.pt    ")
	torch.save(node_features_all, "node_features_all.pt")

	torch.save(H, "H.pt")

# create_and_save_data(1,1)