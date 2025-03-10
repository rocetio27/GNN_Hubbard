import torch
def create_and_save_data(U, t):
	NCn=2
	Size = NCn*NCn
	site_num = 2 # 몇사이트 인가? 

	sing_s1 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
	sing_s1[0]=1
	sing_s1[1]=0

	sing_s2 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
	sing_s2[0]=0
	sing_s2[1]=1


	# 여기까지 각 사이트의 각 스핀의 전자수를 함 써봣다.

	H_list = [[0 for w in range(Size)] for h in range(Size)] #0으로 다 초기화 합시다.

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

	# 여기 까지 스핀 에 대한 호핑 텀 다 만들었다.ㅋ
	# 즉 해밀토니안 다 한 것 ㅋ

	H = torch.tensor(H_list) #텐서로 만들기.

	import numpy as np
	import math
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
	#######################################################################################
	node_features1=torch.tensor([
	        [0   , 0, 0],  # Node 0: [total spin, number of occupation, onsite interaction]
	        [0   , 2, U],  # Node 1
	    ])
	node_features2=torch.tensor([
	        [-1/2 , 1, 0],  # Node 0: [total spin, number of occupation, onsite interaction]
	        [1/2 , 1, 0],  # Node 1
	    ])
	node_features3=torch.tensor([
	        [1/2 , 1, 0],  # Node 0: [total spin, number of occupation, onsite interaction]
	        [-1/2, 1, 0],  # Node 1
	    ])
	node_features4=torch.tensor([
	        [0   , 2, U],  # Node 0: [total spin, number of occupation, onsite interaction]
	        [0   , 0, 0],  # Node 1
	    ])
	#######################################################################################
	edge_attr1=torch.tensor([
	        [0.0, 0.0],
	        [-t, -t],
	    ], dtype=torch.float32)
	edge_attr2=torch.tensor([
	        [0.0 ,-t],
	        [-t, 0.0],
	    ], dtype=torch.float32)
	edge_attr3=torch.tensor([
	        [-t, 0.0],
	        [0.0, -t],
	    ], dtype=torch.float32)
	edge_attr4=torch.tensor([
	        [-t, -t],
	        [0.0, 0.0],
	    ], dtype=torch.float32)
	#######################################################################################
	edge_attr_list = [edge_attr1, edge_attr2, edge_attr3, edge_attr4]
	edge_attr_all = torch.stack(edge_attr_list, dim=0)
	torch.save(edge_attr_all, "edge_attr_all.pt")

	node_features_list = [node_features1, node_features2, node_features3, node_features4]
	node_features_all = torch.stack(node_features_list, dim=0)
	torch.save(node_features_all, "node_features_all.pt")

	edge_index_list = [edge_index1, edge_index2, edge_index3, edge_index4]
	edge_index_all = torch.stack(edge_index_list, dim=0)
	torch.save(edge_index_all, "edge_index_all.pt")

	torch.save(H, "H.pt")