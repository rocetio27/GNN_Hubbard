import torch
def create_and_save_data(U, t):
    NCn=6
    Size = NCn*NCn

    sing_s1 = [0 for h in range(NCn)] #0으로 다 초기화 합시다.
    sing_s1[0]=1
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

    H_list = [[0 for w in range(Size)] for h in range(Size)] #0으로 다 초기화 합시다.

    for la in range(Size):
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

    for la5 in range(NCn):
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

    import numpy as np
    edge_index_array=np.zeros([Size,2,4*2]) # edge_index 3차원으로 가보자 
    for re1 in range(Size):
      edge_index_array[re1] = [
            [0, 1, 1, 2, 2, 3, 3, 0], 
            [1, 0, 2, 1, 3, 2, 0, 3]  
        ]
    edge_index_all = torch.tensor(edge_index_array,dtype=torch.long)

    import math
    node_features_array=np.zeros([Size,4,3]) # 3차원으로 가보자 
    for re2 in range(NCn):
      for re3 in range(NCn):
          node_features_array[re2*NCn + re3] = [
            [sing_s4[re3]*(0.5) + sing_s4[re2]*(-0.5),sing_s4[re3] + sing_s4[re2], math.floor((sing_s4[re3] + sing_s4[re2])/1.5)*U],  # Node 0: [total spin, number of occupation, onsite interaction]
            [sing_s3[re3]*(0.5) + sing_s3[re2]*(-0.5),sing_s3[re3] + sing_s3[re2], math.floor((sing_s3[re3] + sing_s3[re2])/1.5)*U],  # Node 1
            [sing_s2[re3]*(0.5) + sing_s2[re2]*(-0.5),sing_s2[re3] + sing_s2[re2], math.floor((sing_s2[re3] + sing_s2[re2])/1.5)*U],  # Node 2
            [sing_s1[re3]*(0.5) + sing_s1[re2]*(-0.5),sing_s1[re3] + sing_s1[re2], math.floor((sing_s1[re3] + sing_s1[re2])/1.5)*U],  # Node 3
        ]
    node_features_all = torch.tensor(node_features_array,dtype=torch.float32)

    edge_attr_array=np.zeros([Size,4*2,2]) # 3차원으로 가보자 
    for re4 in range(NCn):
      for re5 in range(NCn):
          edge_attr_array[re4*NCn + re5] = [
            [sing_s4[re5]*(sing_s4[re5]-sing_s3[re5])*(-t), sing_s4[re4]*(sing_s4[re4]-sing_s3[re4])*(-t)], #12
            [sing_s3[re5]*(sing_s3[re5]-sing_s4[re5])*(-t), sing_s3[re4]*(sing_s3[re4]-sing_s4[re4])*(-t)], #21
            [sing_s3[re5]*(sing_s3[re5]-sing_s2[re5])*(-t), sing_s3[re4]*(sing_s3[re4]-sing_s2[re4])*(-t)], #23
            [sing_s2[re5]*(sing_s2[re5]-sing_s3[re5])*(-t), sing_s2[re4]*(sing_s2[re4]-sing_s3[re4])*(-t)],  #32
            [sing_s2[re5]*(sing_s2[re5]-sing_s1[re5])*(-t), sing_s2[re4]*(sing_s2[re4]-sing_s1[re4])*(-t)], #34
            [sing_s1[re5]*(sing_s1[re5]-sing_s2[re5])*(-t), sing_s1[re4]*(sing_s1[re4]-sing_s2[re4])*(-t)], #43
            [sing_s1[re5]*(sing_s1[re5]-sing_s4[re5])*(-t), sing_s1[re4]*(sing_s1[re4]-sing_s4[re4])*(-t)],  #41
            [sing_s4[re5]*(sing_s4[re5]-sing_s1[re5])*(-t), sing_s4[re4]*(sing_s4[re4]-sing_s1[re4])*(-t)], #14
        ]
    edge_attr_all = torch.tensor(edge_attr_array,dtype=torch.float32)

    torch.save(edge_attr_all, "edge_attr_all.pt")
    torch.save(node_features_all, "node_features_all.pt")
    torch.save(edge_index_all, "edge_index_all.pt")

    torch.save(H, "H.pt")