import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.nn import Tanh, Sigmoid, Linear, ReLU, Sequential as Seq
from torch_scatter import scatter

dim_hidden=32
alpha_out_dim=5
#---------------------------------------------------------------------------------------------------
# Layer by layer의 Message Passing을 정의
# 엣지 업데이트, 노드 메세지 생성 및 업데이트를 정의하는 클래스
#---------------------------------------------------------------------------------------------------
class MessagePass(MessagePassing):
    def __init__(self,node_feature_dim,edge_attr_dim,gamma_out_dim, edge_update):
        super(MessagePass, self).__init__(aggr='mean')  # Aggregation type: 'add', 'mean', 'max'

        self.node_feature_dim    =node_feature_dim
        self.edge_attr_dim    =edge_attr_dim
        self.edge_update = edge_update
        self.concated_feature_dim=self.node_feature_dim*2+self.edge_attr_dim
        self.message_dim = self.concated_feature_dim//2
        # dim(i-node feature)+dim(j-node feature)+dim(e_ji feature)

        # 엣지의 업데이트를 위한 χ MLP정의
        self.chi = Seq(
            Linear(self.concated_feature_dim, 2*self.concated_feature_dim),
            ReLU(),
            Linear(2*self.concated_feature_dim, 2*self.concated_feature_dim),
            ReLU(),
            Linear(2*self.concated_feature_dim, self.edge_attr_dim)
        )

        # 노드의 메세지 생성을 위한 ϕ MLP 정의
        self.phi = Seq(
            Linear(self.concated_feature_dim, 2*self.concated_feature_dim),
            ReLU(),
            Linear(2*self.concated_feature_dim, 2*self.concated_feature_dim),
            ReLU(),
            Linear(2*self.concated_feature_dim, self.message_dim)
        )

        # 노드의 업데이트를 위한 γ MLP 정의
        self.gamma = Seq(
            Linear(self.node_feature_dim + self.message_dim, 2*(self.node_feature_dim + self.message_dim)),
            ReLU(),
            Linear(2*(self.node_feature_dim + self.message_dim), 2*(self.node_feature_dim + self.message_dim)),
            ReLU(),
            Linear(2*(self.node_feature_dim + self.message_dim), gamma_out_dim)
        )

    def update_edge(self, x, edge_index, edge_attr):
        src, dst=edge_index
        x_i=x[dst]
        x_j=x[src]
        concatenated = torch.cat([x_j, x_i, edge_attr], dim=1)
        return self.chi(concatenated)

    def forward(self, x, edge_index, edge_attr):
        if self.edge_update == True:
            updated_edge_attr = self.update_edge(x, edge_index, edge_attr)
        else:
            updated_edge_attr = edge_attr
        updated_x = self.propagate(edge_index, x=x, edge_attr=updated_edge_attr)
        return updated_edge_attr ,updated_x

    #-------Internal Logic of propagate-----------------------------------
    def message(self, x_i, x_j, edge_attr):
        concatenated = torch.cat([x_j, x_i, edge_attr], dim=1)
        return self.phi(concatenated)

    def update(self, aggr_out, x):
        updated_input = torch.cat([aggr_out, x], dim=1)
        return self.gamma(updated_input)

    #-------Internal Logic of propagate-----------------------------------
    def print_mlps(self):
        """각 MLP(chi, phi, gamma)의 레이어 정보를 출력하는 함수."""
        # print("MLP for chi:")
        # for idx, layer in enumerate(self.chi):
        #     print(f"  Layer {idx}: {layer}")
        
        # print("\nMLP for phi:")
        # for idx, layer in enumerate(self.phi):
        #     print(f"  Layer {idx}: {layer}")
        
        # print("\nMLP for gamma:")
        # for idx, layer in enumerate(self.gamma):
        #     print(f"  Layer {idx}: {layer}")

#---------------------------------------------------------------------------------------------------
# 하나의 spin configuration에 대한 학습을 정의
# 2번의 메세지 패싱과 한번의 풀링 그리고 풀링된 피처를 MLP에 통과시켜서 중간단계의 아웃풋을 뽑아내는 클래스
#---------------------------------------------------------------------------------------------------
class LearningWithinSingleSpinConfiguration(nn.Module):
    def __init__(self, node_feature_dim, edge_attr_dim):
        super(LearningWithinSingleSpinConfiguration, self).__init__()
        self.layer1 = MessagePass(node_feature_dim, edge_attr_dim, node_feature_dim, edge_update=True)
        self.layer1.print_mlps()
        # print(f"############################")
        self.layer2 = MessagePass(node_feature_dim, edge_attr_dim, node_feature_dim, edge_update=True)
        self.layer2.print_mlps()
        # print(f"############################")
        self.pooled_concat_dim=node_feature_dim+edge_attr_dim
        self.alpha = Seq(
            Linear(self.pooled_concat_dim, 2*self.pooled_concat_dim),
            ReLU(),
            # Linear(2*self.pooled_concat_dim, 2*self.pooled_concat_dim),
            # ReLU(),
            Linear(2*self.pooled_concat_dim, alpha_out_dim)
        )

    @staticmethod
    def index_weighted_mean_pool(x, node_batch, ptr):
        """
        x          : 노드 집합 (여러개 그래프 포함).
        node_batch : x의 각 노드가 어느 그래프에 속하는지 index를 return.
        ptr        : ptr[graph_index]=x의 그래프 시작 index
        """
        # local_index = i - ptr[graph] + 1   (1, 2, 3, …)
        batch_node_indexing   = torch.arange(x.size(0), device=x.device)
        local_idx = batch_node_indexing - ptr[node_batch] + 1 #(N,)
        weight    = local_idx.unsqueeze(1).to(x.dtype) #(N,1)

        num   = scatter(x * weight, node_batch, dim=0, reduce='sum')
        return num

    def forward(self, x, edge_index, edge_attr, node_batch, edge_batch, batch):
        edge_attr1, x1 = self.layer1(x, edge_index, edge_attr)
        edge_attr2, x2 = self.layer2(x1, edge_index, edge_attr1)
        node_pooled = self.index_weighted_mean_pool(x2, node_batch, batch.ptr)
        # print(node_pooled)
        edge_pooled = global_mean_pool(edge_attr2, edge_batch)
        # print(edge_pooled)
        pooled_concat = torch.cat([node_pooled, edge_pooled], dim=1)
        estimated_coeff = self.alpha(pooled_concat)
        # print(estimated_coeff)
        return edge_attr2, x2, estimated_coeff

#--------------------------------------------------------------------------------------------
# Spin configuration들 간의 학습을 정의
# 한 batch 내의 각 샘플에서 추출된 파동함수 계수들간의 상관관계를 학습하는 클래스
#--------------------------------------------------------------------------------------------
class LearningBetweenSpinConfigurations(nn.Module):
    def __init__(self, node_feature_dim, edge_attr_dim):
        super(LearningBetweenSpinConfigurations, self).__init__()
        self.instance_LWSSC = LearningWithinSingleSpinConfiguration(node_feature_dim, edge_attr_dim)
        self.layer1 = MessagePass(alpha_out_dim, 1, alpha_out_dim, edge_update=True)
        self.layer2 = MessagePass(alpha_out_dim, 1, alpha_out_dim, edge_update=True)
        self.alpha = Seq(
            Linear(alpha_out_dim, 2*alpha_out_dim),
            ReLU(),
            # Linear(2*alpha_out_dim, 2*alpha_out_dim),
            # ReLU(),
            Linear(2*alpha_out_dim, 1)
        )

    def forward(self, batch, edge_index_LBSC, edge_attr_LBSC):
            updated_edge_attr, updated_x, estimated_coeff = self.instance_LWSSC(
            x=batch.x, 
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            node_batch=batch.batch,
            edge_batch=batch.edge_batch,
            batch=batch
            )
            
            edge_attr_LBSC1 ,estimated_coeff_LBSC1=self.layer1(estimated_coeff      , edge_index_LBSC, edge_attr_LBSC  )
            edge_attr_LBSC2 ,estimated_coeff_LBSC2=self.layer2(estimated_coeff_LBSC1, edge_index_LBSC, edge_attr_LBSC1 )
            # print(estimated_coeff_LBSC2)
            estimated_coeff_LBSC3=self.alpha(estimated_coeff_LBSC2)
            # print(estimated_coeff_LBSC3)
            epsilon = 1e-12  # 0으로 나누는 것을 방지하기 위한 작은 상수
            sum_of_squares = torch.sum(estimated_coeff_LBSC3 ** 2)
            l2_norm = torch.sqrt(sum_of_squares + epsilon)
            estimated_coeff_LBSC3 = estimated_coeff_LBSC3 / l2_norm
            return edge_attr_LBSC1, updated_x, estimated_coeff_LBSC3