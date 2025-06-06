import torch
import os
import module
#--------------------------------------------------------------------------------------------
# 계산 시간 측정
#--------------------------------------------------------------------------------------------
import time
start_time = time.time()  # 시작 시간 기록

#--------------------------------------------------------------------------------------------
# Hubbard Params
#--------------------------------------------------------------------------------------------
U, t=1, 1

#--------------------------------------------------------------------------------------------
# Data creation & save
#--------------------------------------------------------------------------------------------
from generate_hubbard_inputs_Nsite import create_and_save_data
create_and_save_data(2, U, t, "test")

#--------------------------------------------------------------------------------------------
# Batch Making
#--------------------------------------------------------------------------------------------
from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils  import dense_to_sparse
device = torch.device('cpu')
node_features_all = torch.load(os.path.join(module.input_dir, "node_features_all_test.pt"), map_location=device)
edge_attr_all     = torch.load(os.path.join(module.input_dir,     "edge_attr_all_test.pt"), map_location=device)
edge_index_all    = torch.load(os.path.join(module.input_dir,    "edge_index_all_test.pt"), map_location=device)
H                 = torch.load(os.path.join(module.input_dir,                 "H_test.pt"), map_location=device)

graph_dataset = []
num_samples=edge_attr_all.size(0)
for i in range(num_samples):
    ea = edge_attr_all[i]
    ei = edge_index_all[i]
    nf = node_features_all[i]
    data = Data(x=nf, edge_index=ei, edge_attr=ea)
    data.edge_batch = torch.zeros(ea.size(0), dtype=torch.long)
    graph_dataset.append(data)
loader=DataLoader(graph_dataset, batch_size=num_samples, shuffle=False)

#--------------------------------------------------------------------------------------------
# Learning Between Spin Configurations (LBSC) 학습을위한 sample graph의 edge index 및 edge feature 생성
# node는 LWSSC 에서 정의될 것이다.
#--------------------------------------------------------------------------------------------
adj = torch.ones(num_samples, num_samples)
edge_index_LBSC, _= dense_to_sparse(adj)

edge_attr_LBSC = torch.empty((0,))
for i in range(num_samples):
    for j in range(num_samples):
        new_row = H[i, j].unsqueeze(0)  # shape: (1,)
        edge_attr_LBSC=torch.cat([edge_attr_LBSC, new_row], dim=0)
        # edge_attr_LBSC: (num_samples^2)X1 tensor
edge_attr_LBSC = edge_attr_LBSC.unsqueeze(1)
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# 저장된 모델을 통해 평가
#--------------------------------------------------------------------------------------------
from gnn_model import MessagePass, LearningWithinSingleSpinConfiguration, LearningBetweenSpinConfigurations
# 1. 저장 시 사용한 동일한 파라미터로 모델 인스턴스를 생성합니다.
model = LearningBetweenSpinConfigurations(
    node_feature_dim=5,
    edge_attr_dim=2
    )
# 2. 저장된 체크포인트 로드
checkpoint = torch.load(module.checkpoint_name, map_location=torch.device('cpu'))

# 3. 모델 파라미터 로드
model.load_state_dict(checkpoint['model_state_dict'])

# 4. 누적 epoch 정보 로드 (필요한 경우)
total_epochs = checkpoint.get('total_epoch', 0)
print(f"total_epoch: {total_epochs}")

# 5. (옵션) 평가 모드로 전환합니다.
model.eval()

print("Model loaded successfully!")
for step, batch in enumerate(loader):
    print(f"\n================================ BATCH {step} ================================")
    updated_edge_attr, updated_x, estimated_coeff = model(batch, edge_index_LBSC=edge_index_LBSC, edge_attr_LBSC=edge_attr_LBSC)

    # 전파과정을 통해 update된 노드 엣지 피처와 예측된 파동함수 계수 출력.
    unique_sample_indices = batch.batch.unique().tolist()  # 배치 내 샘플 인덱스
    for sample_idx in unique_sample_indices:
        print(f"\n------------ SAMPLE {sample_idx} ------------")
        sample_mask = (batch.batch == sample_idx)  # 해당 샘플의 데이터 선택
        print(f"estimated_wavefunction_coeff:\n {estimated_coeff[sample_idx].item():.5f}")
        # print(f"updated_node_feature:\n {updated_x[sample_idx:sample_idx+2,:]}")
        # print(f"updated_edge_feature:\n {updated_edge_attr[sample_idx:sample_idx+2,:]}")

        # 변분원리에 기반해서 학습 파라미터 수정하기
        # 1. 변분 에너지 계산
        energy=torch.tensor([0.0],dtype=torch.float32)

        # 1.1. <Ψ|Ψ>: 모든 sample(basis)에 대해서 ground state의 내적 계산
        psi_psi=torch.tensor([0.0],dtype=torch.float32)
        for sample_idx in unique_sample_indices:
            sigma_psi=estimated_coeff[sample_idx]
            psi_psi += sigma_psi**2

        # 1.2. ⟨Ψ∣H∣Ψ⟩/⟨Ψ∣Ψ⟩: 계산
        for sample_idx in unique_sample_indices:
            sigma_psi=estimated_coeff[sample_idx]
            sigma_psi_sq=sigma_psi**2
            probability_sigma = sigma_psi / (psi_psi+1e-12)

            E_loc_sigma=torch.tensor([0.0],dtype=torch.float32)
            for sample_idxp in unique_sample_indices:
                sigmap_psi=estimated_coeff[sample_idxp]
                E_loc_sigma+=H[sample_idx,sample_idxp]*(sigmap_psi)
            energy+=probability_sigma*E_loc_sigma

    print(f"Energy: {energy[0].item():.5f} nomalization: {psi_psi.item():.5f}")

#--------------------------------------------------------------------------------------------
# 계산 시간 측정
#--------------------------------------------------------------------------------------------
end_time = time.time()    # 종료 시간 기록
elapsed = end_time - start_time  # 경과 시간 (초 단위)
minutes = int(elapsed // 60)       # 분 단위
seconds = elapsed % 60             # 나머지 초
print("계산시간: {}분 {:.2f}초".format(minutes, seconds))


