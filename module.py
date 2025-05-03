import math
input_dir       = "./inputdata"
checkpoint_name = "true_true.pt"

#training sample grid 변수
n_r = 9                        # radial 방향 그리드 수
n_theta = 9                    # theta 방향 그리드 수 (0 ~ π/2)
r_max = 2.0*math.sqrt(2.0)         # 최대 반지름