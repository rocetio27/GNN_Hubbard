import torch
def create_and_save_data(U, t):
	H=torch.tensor([
	    [U,t,-t,0],
	    [t,0,0,t],
	    [-t,0,0,-t],
	    [0,t,-t,U]
	    ], dtype=torch.float32)

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