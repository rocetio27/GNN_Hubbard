import torch
import os
save_dir="./inputdata"
print(f'CUDA is_available: {torch.cuda.is_available()}')  # Should print True
print(torch.version.cuda)         # Prints the CUDA version PyTorch is built with
print(torch.__version__)
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
from generate_hubbard_inputs_4site import create_and_save_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_4site = []

U_array=[0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75]
t_array=[0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
# U_array=[1]
# t_array=[1]

n_batches=len(U_array)*len(t_array)
print(n_batches)
ut_array = torch.zeros((n_batches, 2))
step=0
for U in U_array:
    for t in t_array:
        if U == 1 and t==1 :
            step_11=step
        ut_array[step, 0] = U
        ut_array[step, 1] = t

        #--------------------------------------------------------------------------------------------
        # Data creation & save
        #--------------------------------------------------------------------------------------------
        create_and_save_data(U, t, H_filename=f"H_{step}.pt")
        step = step + 1
        #--------------------------------------------------------------------------------------------
        # Batch Making
        #-------------------------------------------------------------------------------------------- 
        node_features_all = torch.load(os.path.join(save_dir, "node_features_all.pt"), map_location=device)
        edge_attr_all     = torch.load(os.path.join(save_dir, "edge_attr_all.pt"    ), map_location=device)
        edge_index_all    = torch.load(os.path.join(save_dir, "edge_index_all.pt"   ), map_location=device)


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

# 총 Epoch 수
num_epochs = 200000
interval_save_epoch=100
# 허바드 기저 간 학습 모델 인스턴스 정의
instance_LBSCs=LearningBetweenSpinConfigurations(
    node_feature_dim=5,
    edge_attr_dim=2
    )
instance_LBSCs = instance_LBSCs.to(device)


# 모델 옵티마이저 정의
optimizer = torch.optim.SGD(instance_LBSCs.parameters(), lr=0.005, momentum=0.9)
# optimizer = optim.Adam(instance_LBSCs.parameters(), lr=0.001)

# 모델 불러오기
import os
checkpoint_name = 'true_true.pt'
if os.path.exists(checkpoint_name):
    print("저장된 모델을 불러옵니다.")
    checkpoint = torch.load(checkpoint_name)
    instance_LBSCs.load_state_dict(checkpoint['model_state_dict'])
    total_epochs = checkpoint['total_epoch']
    print(f"누적 학습 에포크: {total_epochs}")
else:
    total_epochs=0



#----------------------------------------------------------------------------------------------------
# Learning Between Spin Configurations (LBSC) 학습을위한 sample graph의 edge index 및 edge feature 생성
# node는 LWSSC 에서 정의될 것이다.
#----------------------------------------------------------------------------------------------------
adj = torch.ones(num_samples, num_samples)

H_batch = torch.zeros(n_batches,num_samples,num_samples)
edge_index_LBSC, _= dense_to_sparse(adj)
print(edge_index_LBSC)

edge_index_LBSC = edge_index_LBSC.to(device)
H_batch = H_batch.to(device)

edge_attr_LBSC_batch = torch.zeros((n_batches, num_samples**2, 1))
edge_attr_LBSC_batch = edge_attr_LBSC_batch.to(device)

for step, batch in enumerate(loader):
    batch = batch.to(device)
    # create_and_save_data(ut_array[step, 0].item(), ut_array[step, 1].item())
    H = torch.load(os.path.join(save_dir, f"H_{step}.pt"))
    edge_attr_LBSC = torch.empty((0,))
    for i in range(num_samples):
        for j in range(num_samples):
            new_row = H[i, j].unsqueeze(0)  # shape: (1,)
            edge_attr_LBSC=torch.cat([edge_attr_LBSC, new_row], dim=0) # edge_attr_LBSC: (num_samples^2)X1 tensor
    edge_attr_LBSC = edge_attr_LBSC.unsqueeze(1)
    edge_attr_LBSC_batch[step]=edge_attr_LBSC
    H_batch[step,:,:]=H

#--------------------------------------------------------------------------------------------
# GNN Variational Optimization
#--------------------------------------------------------------------------------------------
for epoch in range(num_epochs):
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        updated_edge_attr, updated_x, estimated_coeff = instance_LBSCs(batch, edge_index_LBSC=edge_index_LBSC, edge_attr_LBSC=edge_attr_LBSC_batch[step,:])
        if step == step_11 :
            print(f"\n=============== BATCH {step} ================================")
        unique_sample_indices = batch.batch.unique().tolist()  # 배치 내 샘플 인덱스

        # for sample_idx in unique_sample_indices:
        #     print(f"\n--------------- SAMPLE {sample_idx} ---------------")
        #     sample_mask = (batch.batch == sample_idx)  # 해당 샘플의 데이터 선택
        #     print(f"estimated_wavefunction_coeff.:\n {estimated_coeff[sample_idx].item():.5f}")

        ##################################Variational Opimization##################################
        # 1. 변분 에너지 계산

        # 1.1. <Ψ|Ψ>: 모든 sample(basis)에 대해서 ground state의 내적 계산
        psi_psi = (estimated_coeff**2).sum()

        # 1.2. ⟨Ψ∣H∣Ψ⟩/⟨Ψ∣Ψ⟩: 계산
        probability_sigma_vec = (estimated_coeff[unique_sample_indices] / (psi_psi + 1e-12)).view(1, -1) # (1 X n)
        H_sub = H_batch[step, unique_sample_indices][:, unique_sample_indices]                           # (n X n)
        estimated_coeff_vec = estimated_coeff[unique_sample_indices].view(-1, 1)                         # (n X 1)
        energy = (probability_sigma_vec @ H_sub @ estimated_coeff_vec).squeeze()

        # 2. loss function 정의
        loss = energy

        # 3. 가중치 업데이트
        optimizer.zero_grad()  # 그래디언트 초기화
        loss.backward()        # 역전파 수행
        optimizer.step()       # 가중치 업데이트
        if step == step_11 :
            print(f"Epoch {epoch+1}/{num_epochs}, Loss(Energy): {loss.item():.5f} Normalization: {psi_psi.item():.5f}")

    if epoch % interval_save_epoch == 1:
        total_epochs = total_epochs + interval_save_epoch  # 이번 학습 횟수를 누적
        checkpoint = {
            'total_epoch': total_epochs,
            'model_state_dict': instance_LBSCs.state_dict()
        }
        print('model has sucessfully saved')
        torch.save(checkpoint, checkpoint_name)


#--------------------------------------------------------------------------------------------
# 계산 시간 측정
#--------------------------------------------------------------------------------------------
end_time = time.time()             # 종료 시간 기록
elapsed = end_time - start_time    # 경과 시간 (초 단위)
minutes = int(elapsed // 60)       # 분 단위
seconds = elapsed % 60             # 나머지 초
print("계산시간: {}분 {:.2f}초".format(minutes, seconds))
