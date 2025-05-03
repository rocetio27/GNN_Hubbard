
import os, torch, math
import module
# Cuation: Allowed duplicating OpenMP runtime (temporary workaround)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils  import dense_to_sparse

from generate_hubbard_inputs_Nsite  import create_and_save_data
from gnn_model                      import LearningBetweenSpinConfigurations

def draw(epoch_num):
    # 1) 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2) 오버레이할 트레이닝 샘플 포인트 생성
    n_r     = module.n_r        # radial grid 개수
    n_theta = module.n_theta    # angular grid 개수 (0 ~ π/2)
    r_max   = module.r_max      # 최대 반지름

    train_samples = []

    for i_r in range(n_r):
        r = r_max * i_r / (n_r-1)
        for i_theta in range(1,n_theta-1):
            theta = (math.pi / 2) * i_theta / (n_theta - 1)

            U = r * math.cos(theta)
            t = r * math.sin(theta)

            if abs(U) < 1e-8 and abs(t) < 1e-8:
                continue  # (0,0)은 제외

            train_samples.append((U, t))

    train_us = [pt[0] for pt in train_samples]
    train_ts = [pt[1] for pt in train_samples]

    # 3) 그리드 스캔 파라미터
    U_min_plot, U_max_plot = 0.0, 8.0
    t_min_plot, t_max_plot = 0.0, 8.0

    grid            = 24
    step            = U_max_plot / grid      # 시작점을 step 만큼 띄워 0 제외
    us              = torch.linspace(step, U_max_plot, grid, device=device)
    ts              = torch.linspace(step, t_max_plot, grid, device=device)
    pred_energy_mat = torch.zeros((grid, grid), device=device)
    gs_energy_mat   = torch.zeros((grid, grid), device=device)

    # 4) 학습된 모델 로드
    model = LearningBetweenSpinConfigurations(node_feature_dim=5, edge_attr_dim=2).to(device)
    ckpt  = torch.load(f'{epoch_num}_true_true.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 5) 그리드 스캔
    for iu, U in enumerate(us):
        for it, t in enumerate(ts):

            # 5.1) 노드/엣지 피처 로드
            node_path       = os.path.join(module.input_dir, f"node_features_all_{U:.2f}_{t:.2f}.pt")
            edge_attr_path  = os.path.join(module.input_dir,     f"edge_attr_all_{U:.2f}_{t:.2f}.pt")
            edge_index_path = os.path.join(module.input_dir,    f"edge_index_all_{U:.2f}_{t:.2f}.pt")
            H_path          = os.path.join(module.input_dir,                 f"H_{U:.2f}_{t:.2f}.pt")
            if  not (os.path.exists(node_path) and os.path.exists(edge_attr_path) and os.path.exists(edge_index_path) and os.path.exists(H_path)):
                create_and_save_data(4, U.item(), t.item(), f"{iu:.2f}_{it:.2f}")

            node_features_all = torch.load(os.path.join(module.input_dir, f"node_features_all_{U:.2f}_{t:.2f}.pt"), map_location=device)
            edge_attr_all     = torch.load(os.path.join(module.input_dir,     f"edge_attr_all_{U:.2f}_{t:.2f}.pt"), map_location=device)
            edge_index_all    = torch.load(os.path.join(module.input_dir,    f"edge_index_all_{U:.2f}_{t:.2f}.pt"), map_location=device)
            H                 = torch.load(os.path.join(module.input_dir,                 f"H_{U:.2f}_{t:.2f}.pt"), map_location=device)

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
    error_mat = 100 * (pred_energy_mat - gs_energy_mat) / (gs_energy_mat.abs() + 1e-12)
    error_mat = error_mat.clamp(0, 5.0).cpu().numpy()  # 퍼센트 기준으로 클리핑

    # 7) 컬러맵 & 노멀라이즈
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    vmin, vmax = 0.0, 5.0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # 8) 플롯
    plt.figure(figsize=(8.5,7), constrained_layout=True)
    im = plt.imshow(
        error_mat,
        origin='lower',
        extent=[t_min_plot, t_max_plot, U_min_plot, U_max_plot],
        aspect='auto',
        cmap='Reds',
        norm=norm
    )

    FontSize=18
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels([f"{t:.0f}%" for t in [0, 1, 2, 3, 4, 5]], fontsize=(FontSize-2))
    cbar.set_label('Relative error (%)', fontsize=FontSize)

    # 9) 트레이닝 샘플 오버레이
    plt.scatter(
        train_ts, train_us,
        c='white', s=60, marker='o',
        edgecolors='black', linewidths=1,
        label='Training samples'
    )

    legend = plt.legend(loc='upper left', frameon=False, fontsize=FontSize)
    for text in legend.get_texts():
        text.set_color("white")
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()
        ])

    plt.xlabel('t (nearest neighbor hopping)', fontsize=FontSize)
    plt.ylabel('U (on-site interaction)', fontsize=FontSize)
    plt.title('Predicted Variational Energy vs Exact Ground State Energy\nRelative Error', fontsize=FontSize)
    plt.xticks(fontsize=FontSize)
    plt.yticks(fontsize=FontSize)
    plt.savefig(f'{epoch_num}.svg')
    # plt.show()
# draw(750)