import time
import glob
import figure_plot
seen_epochs = set()
# seen_epochs = set([
#     int(f.split('_')[0]) for f in glob.glob("*_true_true.pt")
# ])

while True:
    # 저장된 파일 목록 중, *_true_true.pt 형식인 것만 필터링
    pt_files = glob.glob("*_true_true.pt")

    # epoch 번호 추출 (예: '750_true_true.pt' → 750)
    epoch_nums = sorted([
        int(f.split('_')[0]) for f in pt_files
        if f.endswith("_true_true.pt")
    ])

    for epoch_num in epoch_nums:
        if epoch_num not in seen_epochs:
            print(f"New epoch detected: {epoch_num}")
            seen_epochs.add(epoch_num)

            try:
                figure_plot.draw(epoch_num)
            except Exception as e:
                print(f"Failed to process epoch {epoch_num}: {e}")

    time.sleep(30)  # 30초마다 폴더를 확인