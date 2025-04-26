import os, torch
# Cuation: Allowed duplicating OpenMP runtime (temporary workaround)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils  import dense_to_sparse

from generate_hubbard_inputs_4site  import create_and_save_data
from gnn_model                      import LearningBetweenSpinConfigurations
from hubbard_utils                  import cal_mat_el

# 0) 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) 플롯 범위 매개변수
U_min_plot, U_max_plot = 0.0, 8.0
t_min_plot, t_max_plot = 0.0, 8.0

# 2) 트레이닝 샘플 포인트 생성
U_array = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75]
t_array = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
train_samples = [(U, t) for U in U_array for t in t_array]
train_us = [pt[0] for pt in train_samples]
train_ts = [pt[1] for pt in train_samples]

# 3) 그리드 스캔 파라미터
site_num        = 24
grid            = 24
step            = U_max_plot / grid      # 시작점을 step 만큼 띄워 0 제외
us              = torch.linspace(step, U_max_plot, grid, device=device)
ts              = torch.linspace(step, t_max_plot, grid, device=device)
pred_energy_mat = torch.zeros((grid, grid), device=device)
gs_energy_mat   = torch.zeros((grid, grid), device=device)

# 4) 학습된 모델 로드
model = LearningBetweenSpinConfigurations(node_feature_dim=5, edge_attr_dim=2).to(device)
ckpt  = torch.load('true_true.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 5) 그리드 스캔
for iu, U in enumerate(us):
    for it, t in enumerate(ts):
        # 5.1) 샘플 데이터 생성
        create_and_save_data(U.item(), t.item(), "H_tmp.pt")

        # 5.2) 노드/엣지 피처 로드
        node_features_all = torch.load(os.path.join(save_dir, "node_features_all.pt"), map_location=device)
        edge_attr_all     = torch.load(os.path.join(save_dir, "edge_attr_all.pt"    ), map_location=device)
        edge_index_all    = torch.load(os.path.join(save_dir, "edge_index_all.pt"   ), map_location=device)

        # 5.3) 해밀토니언 생성 & GPU 배치
        H = torch.load(os.path.join("./inputdata","H_tmp.pt")).float().to(device)
        basis_num = edge_attr_all.size(0)

        # 5.4) DataLoader 준비
        dataset = []
        for nf, ea, ei in zip(node_features_all, edge_attr_all, edge_index_all):
            d = Data(x=nf, edge_index=ei, edge_attr=ea)
            d.edge_batch = torch.zeros(ea.size(0), dtype=torch.long, device=device)
            dataset.append(d)
        loader = DataLoader(dataset, batch_size=basis_num, shuffle=False)
        batch  = next(iter(loader))

        # 5.5) LBSC용 완전 그래프 인덱스
        adj               = torch.ones(basis_num, basis_num, device=device)
        edge_index_LBSC,_ = dense_to_sparse(adj)

        # 5.6) 변분 에너지 계산 (test.py 블록 그대로)
        with torch.no_grad():
            edge_attr_LBSC = H.flatten().unsqueeze(1)
            _, _, est_coeff = model(batch,
                                    edge_index_LBSC=edge_index_LBSC,
                                    edge_attr_LBSC=edge_attr_LBSC)

            psi_psi = (est_coeff**2).sum()
            idx     = torch.arange(basis_num, device=device)

            prob_vec  = (est_coeff[idx] / (psi_psi + 1e-12)).view(1, -1)
            H_sub      = H[idx][:, idx]
            coeff_vec  = est_coeff[idx].view(-1, 1)
            energy_var = (prob_vec @ H_sub @ coeff_vec).squeeze()
            pred_energy = energy_var.item()

        # 5.7) 정확한 GS 에너지
        gs_energy = torch.linalg.eigh(H).eigenvalues.min().item()

        # 5.8) 결과 저장
        pred_energy_mat[iu, it] = pred_energy
        gs_energy_mat[iu, it]   = gs_energy
        print(f'U={U:.2f}, t={t:.2f} → pred {pred_energy:.4f}, exact {gs_energy:.4f}')

# 6) 상대 오차 계산 & 클리핑
error_mat = (pred_energy_mat - gs_energy_mat) / (gs_energy_mat.abs() + 1e-12)
error_mat = error_mat.clamp(-0.05, 0.05).cpu().numpy()

# 7) 컬러맵 & 노르멀라이즈
vmin, vmax = -0.05, 0.05
norm       = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# 8) 플롯
plt.figure(figsize=(6,5))
im = plt.imshow(
    error_mat,
    origin='lower',
    extent=[t_min_plot, t_max_plot, U_min_plot, U_max_plot],
    aspect='auto',
    cmap='bwr',
    norm=norm
)
cbar = plt.colorbar(im, ticks=[vmin, -0.01, 0.0, 0.01, vmax])
cbar.ax.set_yticklabels([f"{t*100:.0f}%" for t in [vmin, -0.01, 0, 0.01, vmax]])
cbar.set_label('Relative error (%)')

# 9) 트레이닝 샘플 오버레이
plt.scatter(
    train_ts, train_us,
    c='k', s=40, marker='o',
    edgecolors='white', linewidths=0.5,
    label='training samples'
)

plt.xlabel('t')
plt.ylabel('U')
plt.title('Predicted Variational Energy vs Exact Ground State Energy — Relative Error')
plt.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig('error_map_with_samples.png', dpi=300)
plt.show()
