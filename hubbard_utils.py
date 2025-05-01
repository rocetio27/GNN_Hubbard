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
		if i_spin_anni==0: # If i_spin_anni corresponds to down-spin then we need to count additional up-spin site above the down-spin site.
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
			temp_node_features[i_node_crea,i_spin_anni+1]=1
			sign=(-1)**(num_occ1+num_occ2)
	return sign, temp_node_features

def cal_mat_el(i,j,node_features_list,site_num,U,t):
	sample_i_node_features=node_features_list[i].clone()
	sample_j_node_features=node_features_list[j].clone()
	if i == j:
		count_doublon = (sample_j_node_features[:,4]!=0).sum().item()
		mat_el        = count_doublon * U
		return mat_el
	else:
		mat_el = 0
		bra    = sample_i_node_features[:,[1,2]]
		for i_node in range(site_num):
			for i_spin in [0,1]:
				if sample_j_node_features[i_node, i_spin+1] == 1 or sample_j_node_features[(i_node+1)%site_num, i_spin+1] == 1:

					sign, ket_full = cal_sign((i_node+1)%site_num, i_node, i_spin, sample_j_node_features, site_num)
					ket            = ket_full[:,[1,2]]
					if torch.equal(bra, ket):
						mat_el = mat_el + sign*(-t)

					sign, ket_full = cal_sign(i_node, (i_node+1)%site_num, i_spin, sample_j_node_features, site_num)
					ket            = ket_full[:,[1,2]]
					if torch.equal(bra, ket):
						mat_el = mat_el + sign*(-t)
		if site_num	== 2:
			return mat_el/2
		else:
			return mat_el