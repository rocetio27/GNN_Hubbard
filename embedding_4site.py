import torch
import math
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
    temp_node_features[i_node_anni,i_spin_anni+1]=0 # This results from a_(i,σ)a†_(i,σ)=1-a†_(i,σ)a_(i,σ)
      
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
    site_num=4
    NCn=6 # of possible up(down) spin configurations
    basis_num = NCn*NCn # of basis

    sing_s1 = [0 for h in range(NCn)] #0으로 다 초기화 합시다. $single_spin & site 1 & index=basis
    sing_s1[0]=1 # occupation at site1
    sing_s1[1]=1
    sing_s1[2]=0
    sing_s1[3]=1
    sing_s1[4]=0
    sing_s1[5]=0

    sing_s2 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
    sing_s2[0]=1
    sing_s2[1]=0
    sing_s2[2]=1
    sing_s2[3]=0
    sing_s2[4]=1
    sing_s2[5]=0

    sing_s3 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
    sing_s3[0]=0
    sing_s3[1]=1
    sing_s3[2]=1
    sing_s3[3]=0
    sing_s3[4]=0
    sing_s3[5]=1

    sing_s4 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
    sing_s4[0]=0
    sing_s4[1]=0
    sing_s4[2]=0
    sing_s4[3]=1
    sing_s4[4]=1
    sing_s4[5]=1

    # 여기까지 각 사이트의 각 스핀의 전자수를 함 써봣다.

    H_list = [[0 for w in range(basis_num)] for h in range(basis_num)] #0으로 다 초기화 합시다.

    for la in range(basis_num): # laN은 dummy 아무 의미 없음.
      H_list[la][la]=U # 일단 모든 H의 대각 요소를 U로 초기화 해준다.

    for la2 in range(NCn):
      H_list[(NCn+1)*la2][(NCn+1)*la2]=2*U  # 그 중에서 NCn개는 2*U이다.

    for la3 in range(NCn):
      H_list[(NCn-1)*(la3+1)][(NCn-1)*(la3+1)]=0   # 그 중에서 NCn개는 0이다.

    # 여기까지 U에 관한 해밀토니안 다 만들었다.

    # spin up 에 대한 hopping 해밀토니안 만들기.

    for la4 in range(NCn):
      H_list[NCn*la4][1+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ1> ->|Ψ2>
      H_list[NCn*la4][4+NCn*la4]=-t*(-1)**(sing_s2[0]+sing_s3[0]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ1> ->|Ψ5>
      H_list[NCn*la4+1][0+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ2> ->|Ψ1>
      H_list[NCn*la4+1][2+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ2> ->|Ψ3>
      H_list[NCn*la4+1][3+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ2> ->|Ψ4>
      H_list[NCn*la4+1][5+NCn*la4]=-t*(-1)**(sing_s2[1]+sing_s3[1]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ2> ->|Ψ6>
      H_list[NCn*la4+2][1+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ3> ->|Ψ2>
      H_list[NCn*la4+2][4+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ3> ->|Ψ5>
      H_list[NCn*la4+3][1+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ4> ->|Ψ2>
      H_list[NCn*la4+3][4+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ4> ->|Ψ5>
      H_list[NCn*la4+4][0+NCn*la4]=-t*(-1)**(sing_s2[4]+sing_s3[4]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ5> ->|Ψ1>
      H_list[NCn*la4+4][2+NCn*la4]=-t*(-1)**(sing_s4[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ5> ->|Ψ3>
      H_list[NCn*la4+4][3+NCn*la4]=-t*(-1)**(sing_s2[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ5> ->|Ψ4>
      H_list[NCn*la4+4][5+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ5> ->|Ψ6>
      H_list[NCn*la4+5][1+NCn*la4]=-t*(-1)**(sing_s2[5]+sing_s3[5]+sing_s2[la4]+sing_s3[la4]+sing_s4[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ6> ->|Ψ2>
      H_list[NCn*la4+5][4+NCn*la4]=-t*(-1)**(sing_s3[la4])  # 이거 이제 ppt11 페이지 참고. |Ψ6> ->|Ψ5>

    # spin down 에 대한 hopping 해밀토니안 만들기.

    for la5 in range(NCn): #
      H_list[la5][6+la5]=-t*(-1)**(sing_s2[la5])  # 이거 이제 ppt12 페이지 참고. |Ψ1> ->|Ψ7>
      H_list[la5][24+la5]=-t*(-1)**(sing_s2[0]+sing_s3[0]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5])  # 이거 이제 ppt12 페이지 참고. |Ψ1> ->|Ψ25>
      H_list[la5+NCn*1][0+la5]=-t*(-1)**(sing_s2[la5])  # 이거 이제 ppt12 페이지 참고. |Ψ7> ->|Ψ1>
      H_list[la5+NCn*1][12+la5]=-t*(-1)**(sing_s1[la5])  # 이거 이제 ppt12 페이지 참고. |Ψ7> ->|Ψ13>
      H_list[la5+NCn*1][18+la5]=-t*(-1)**(sing_s3[la5])  # 이거 이제 ppt12 페이지 참고. |Ψ7> ->|Ψ19>
      H_list[la5+NCn*1][30+la5]=-t*(-1)**(sing_s2[1]+sing_s3[1]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5])  # 이거 이제 ppt12 페이지 참고. |Ψ7> ->|Ψ31>
      H_list[la5+NCn*2][6+la5]=-t*(-1)**(sing_s1[la5])  # 이 |Ψ13> ->|Ψ7>
      H_list[la5+NCn*2][24+la5]=-t*(-1)**(sing_s3[la5])  # 이 |Ψ13> ->|Ψ25>
      H_list[la5+NCn*3][6+la5]=-t*(-1)**(sing_s3[la5])  # 이 |Ψ19> ->|Ψ7>
      H_list[la5+NCn*3][24+la5]=-t*(-1)**(sing_s1[la5])  # 이 |Ψ19> ->|Ψ25>
      H_list[la5+NCn*4][0+la5]=-t*(-1)**(sing_s2[4]+sing_s3[4]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5])  # 이 |Ψ25> ->|Ψ1>
      H_list[la5+NCn*4][12+la5]=-t*(-1)**(sing_s3[la5])  # 이 |Ψ25> ->|Ψ13>
      H_list[la5+NCn*4][18+la5]=-t*(-1)**(sing_s1[la5])  # 이 |Ψ25> ->|Ψ19>
      H_list[la5+NCn*4][30+la5]=-t*(-1)**(sing_s2[la5])  # 이 |Ψ25> ->|Ψ31>
      H_list[la5+NCn*5][6+la5]=-t*(-1)**(sing_s2[5]+sing_s3[5]+sing_s1[la5]+sing_s2[la5]+sing_s3[la5])  # 이 |Ψ31> ->|Ψ7>
      H_list[la5+NCn*5][24+la5]=-t*(-1)**(sing_s2[la5])  # 이 |Ψ31> ->|Ψ25>

    H = torch.tensor(H_list)
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
          sign=cal_sign(i_node_crea, i_node_anni, i_spin_anni, temp_node_features, site_num)
          temp_list.append(-t*sign)
        temp_tensor = torch.tensor(temp_list, dtype=torch.float32)
        tensor_list.append(temp_tensor)
      edge_attr_list.append(torch.stack(tensor_list, dim=0))
    ##########################################################################################
    edge_index_all    = torch.stack(edge_index_list   , dim=0)
    edge_attr_all     = torch.stack(edge_attr_list    , dim=0)
    node_features_all = torch.stack(node_features_list, dim=0)

    # for i, tensor in enumerate(edge_attr_list):
    #   print(f"Sample {i}:")
    #   print(tensor)
    #   print()

    torch.save(edge_attr_all, "edge_attr_all.pt")
    torch.save(node_features_all, "node_features_all.pt")
    torch.save(edge_index_all, "edge_index_all.pt")
    torch.save(H, "H.pt")

# create_and_save_data(1,1)

    ###########################################################################################
    # Code Backup
    # edge_attr_array=np.zeros([basis_num,4*2,2]) # 3차원으로 가보자 
    # for re4 in range(NCn): # spin down
    #   for re5 in range(NCn): # spin up
    #       edge_attr_array[re4*NCn + re5] = [ #[up_spin,down_spin]
    #         [sing_s4[re5]*(sing_s4[re5]-sing_s3[re5])*(-t), sing_s4[re4]*(sing_s4[re4]-sing_s3[re4])*(-t)], #e_12 (j notation) (sN 는 G-notation 역순)
    #         [sing_s3[re5]*(sing_s3[re5]-sing_s4[re5])*(-t), sing_s3[re4]*(sing_s3[re4]-sing_s4[re4])*(-t)], #e_21

    #         [sing_s3[re5]*(sing_s3[re5]-sing_s2[re5])*(-t), sing_s3[re4]*(sing_s3[re4]-sing_s2[re4])*(-t)], #e_23
    #         [sing_s2[re5]*(sing_s2[re5]-sing_s3[re5])*(-t), sing_s2[re4]*(sing_s2[re4]-sing_s3[re4])*(-t)], #e_32

    #         [sing_s2[re5]*(sing_s2[re5]-sing_s1[re5])*(-t), sing_s2[re4]*(sing_s2[re4]-sing_s1[re4])*(-t)], #e_34
    #         [sing_s1[re5]*(sing_s1[re5]-sing_s2[re5])*(-t), sing_s1[re4]*(sing_s1[re4]-sing_s2[re4])*(-t)], #e_43

    #         [sing_s1[re5]*(sing_s1[re5]-sing_s4[re5])*(-t), sing_s1[re4]*(sing_s1[re4]-sing_s4[re4])*(-t)], #e_41
    #         [sing_s4[re5]*(sing_s4[re5]-sing_s1[re5])*(-t), sing_s4[re4]*(sing_s4[re4]-sing_s1[re4])*(-t)], #e_14
    #     ]
    # edge_attr_all = torch.tensor(edge_attr_array,dtype=torch.float32)