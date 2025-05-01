import torch

#--------------------------------------------------------------------------------------------
# 계산 시간 측정
#--------------------------------------------------------------------------------------------
import time
start_time = time.time()  # 시작 시간 기록

#--------------------------------------------------------------------------------------------
# Hubbard Params
#--------------------------------------------------------------------------------------------
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from embedding_4site import create_and_save_data

dataset_4site = []

U_array=[0, 0.25, 0.5, 0.75, 1, 1.25]
t_array=[0, 0.25, 0.5, 0.75, 1, 1.25]
n_batches=len(U_array)*len(t_array)
print(n_batches)
ut_array = torch.zeros((n_batches, 2))
count=0
for U in U_array:
    for t in t_array:
        ut_array[count, 0] = U
        ut_array[count, 1] = t
        count += 1

        #--------------------------------------------------------------------------------------------
        # Data creation & save
        #--------------------------------------------------------------------------------------------
        create_and_save_data(U, t)

        #--------------------------------------------------------------------------------------------
        # Batch Making
        #-------------------------------------------------------------------------------------------- 
        node_features_all = torch.load("node_features_all.pt", weights_only=True)
        edge_attr_all = torch.load("edge_attr_all.pt", weights_only=True)
        edge_index_all = torch.load("edge_index_all.pt", weights_only=True)

        # dataset_4site = []
        num_samples=edge_attr_all.size(0)
        for i in range(num_samples):
            ea = edge_attr_all[i]
            ei = edge_index_all[i]
            nf = node_features_all[i]
            data = Data(x=nf, edge_index=ei, edge_attr=ea)
            data.edge_batch = torch.zeros(ea.size(0), dtype=torch.long)
            dataset_4site.append(data)
loader=DataLoader(dataset_4site, batch_size=num_samples, shuffle=False)

#--------------------------------------------------------------------------------------------
# 학습 코드, GNN model은 gnn_model.py에 저장되어있음.
#--------------------------------------------------------------------------------------------
from gnn_model import MessagePass, LearningWithinSingleSpinConfiguration, LearningBetweenSpinConfigurations
import torch.optim as optim
# 샘플별 파동함수 계수 및 에너지 저장을 위한 list 정의
num_epochs = 50000                                              # 총 Epoch 수
energy_real_history = []                                         # 전체 에너지 기록
wavefunction_real_history = {i: [] for i in range(num_samples)}  # 샘플별 실수부 기록

# 모델 인스턴스 정의
instance_LBSCs=LearningBetweenSpinConfigurations(
    node_feature_dim=5,
    edge_attr_dim=2
    )

# 모델 옵티마이저 정의
# optimizer = torch.optim.SGD(instance_LBSCs.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(instance_LBSCs.parameters(), lr=0.002)

# 모델 불러오기
# import os
# checkpoint_path = 'hubbard_4site2.pt'
# if os.path.exists(checkpoint_path):
#     print("저장된 모델을 불러옵니다.")
#     instance_LBSCs.load_state_dict(torch.load(checkpoint_path))
# else:
#     print("저장된 모델 파일이 없습니다. 계산을 종료합니다.")
#     quit()


# 모델 파이프라인 출력
# print(f"\n",instance_LBSCs)

#--------------------------------------------------------------------------------------------
# Learning Between Spin Configurations (LBSC) 학습을위한 sample graph의 edge index 및 edge feature 생성
# node는 LWSSC 에서 정의될 것이다.
#--------------------------------------------------------------------------------------------
adj = torch.ones(num_samples, num_samples)
H_batch = torch.zeros(n_batches,num_samples,num_samples)
edge_index_LBSC, _= dense_to_sparse(adj)
print(edge_index_LBSC)

edge_attr_LBSC_batch = torch.zeros((n_batches, num_samples**2, 1))
for step, batch in enumerate(loader):
    create_and_save_data(ut_array[step, 0].item(), ut_array[step, 1].item())
    H = torch.load("H.pt", weights_only=True)
    edge_attr_LBSC = torch.empty((0,))
    for i in range(num_samples):
        for j in range(num_samples):
            new_row = H[i, j].unsqueeze(0)  # shape: (1,)
            edge_attr_LBSC=torch.cat([edge_attr_LBSC, new_row], dim=0) # edge_attr_LBSC: (num_samples^2)X1 tensor
    edge_attr_LBSC = edge_attr_LBSC.unsqueeze(1)
    edge_attr_LBSC_batch[step]=edge_attr_LBSC
    H_batch[step,:,:]=H


# # >>>>>>>>>>>>>모델 옵티마이제이션
for epoch in range(num_epochs):
    for step, batch in enumerate(loader):
        H=H_batch[step,:,:]
        updated_edge_attr, updated_x, estimated_coeff = instance_LBSCs(batch, edge_index_LBSC=edge_index_LBSC, edge_attr_LBSC=edge_attr_LBSC_batch[step,:])
        print(f"\n=============== BATCH {step} ================================")
        unique_sample_indices = batch.batch.unique().tolist()  # 배치 내 샘플 인덱스
        for sample_idx in unique_sample_indices:
            print(f"\n--------------- SAMPLE {sample_idx} ---------------")
            sample_mask = (batch.batch == sample_idx)  # 해당 샘플의 데이터 선택
            print(f"estimated_wavefunction_coeff:\n {estimated_coeff[sample_idx].item():.5f}")

        # 변분원리에 기반해서 학습 파라미터 수정하기
        # 1. 변분 에너지 계산
        energy=0.0

        # 1.1. <Ψ|Ψ>: 모든 sample(basis)에 대해서 ground state의 내적 계산
        psi_psi = (estimated_coeff ** 2).sum()

        # 1.2. ⟨Ψ∣H∣Ψ⟩/⟨Ψ∣Ψ⟩: 계산
        for sample_idx in unique_sample_indices:
            sigma_psi=estimated_coeff[sample_idx]
            sigma_psi_sq=sigma_psi**2
            probability_sigma = sigma_psi / (psi_psi+1e-12)

            E_loc_sigma=0.0
            for sample_idxp in unique_sample_indices:
                sigmap_psi=estimated_coeff[sample_idxp]
                E_loc_sigma+=H[sample_idx,sample_idxp]*(sigmap_psi)
            energy+=probability_sigma*E_loc_sigma

        # 2. loss function 정의
        loss = energy

        # 3. 가중치 업데이트
        optimizer.zero_grad()  # 그래디언트 초기화
        loss.backward()        # 역전파 수행
        optimizer.step()       # 가중치 업데이트

        # 한 Epoch가 끝날 때 총 에너지 저장
        energy_real_history.append(energy[0].item())
        for sample_idx in unique_sample_indices:
            sigma_psi = estimated_coeff[sample_idx]
            real_coeff = sigma_psi.item()  # 실수부

            # 파동함수 계수 저장
            wavefunction_real_history[sample_idx].append(real_coeff)
            # wavefunction_imag_history[sample_idx].append(imag_coeff)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss(Energy): {loss.item():.5f} Normalization: {psi_psi.item():.5f}")

torch.save(instance_LBSCs.state_dict(), 'hubbard_4site2.pt')


#--------------------------------------------------------------------------------------------
# 계산 시간 측정
#--------------------------------------------------------------------------------------------
end_time = time.time()    # 종료 시간 기록
elapsed = end_time - start_time  # 경과 시간 (초 단위)
minutes = int(elapsed // 60)       # 분 단위
seconds = elapsed % 60             # 나머지 초
print("계산시간: {}분 {:.2f}초".format(minutes, seconds))
