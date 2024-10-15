import matplotlib.pyplot as plt

def plot_k_fold_results(file_path, output_image_path, k=10):

    all_epochs = []
    all_train_losses = []
    all_val_accuracies = []
    test_accuracies = []

    current_fold_epochs = []
    current_fold_train_losses = []
    current_fold_val_accuracies = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('Epoch') and 'Train_loss' in line:
                parts = line.split()
                epoch_num = int(parts[1].strip('[]'))
                train_loss = float(parts[3])
                current_fold_epochs.append(epoch_num)
                current_fold_train_losses.append(train_loss)
            elif line.startswith('Epoch') and 'Test results' in line:
                parts = line.split()
                val_acc = float(parts[-1])
                current_fold_val_accuracies.append(val_acc)
            elif line.startswith('Split') and 'acc_test' in line:
                # 当前折结束，记录数据
                if current_fold_epochs:
                    all_epochs.append(current_fold_epochs)
                    all_train_losses.append(current_fold_train_losses)
                    all_val_accuracies.append(current_fold_val_accuracies)
                    current_fold_epochs = []
                    current_fold_train_losses = []
                    current_fold_val_accuracies = []
                # 记录测试准确率
                test_acc = float(line.split()[-1])
                test_accuracies.append(test_acc)

    # 绘图
    plt.figure(figsize=(14, 7))

    for i in range(k):
        plt.subplot(1, 2, 1)
        plt.plot(all_epochs[i], all_train_losses[i], label=f'split {i+1} loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(all_epochs[i], all_val_accuracies[i], label=f'split {i+1} acc_val')
        plt.xlabel('Epoch')
        plt.ylabel('acc')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    # 打印并显示每个 Split 的测试准确率
    for i, acc in enumerate(test_accuracies):
        print(f"折 {i+1} 的测试准确率: {acc}")

    # 保存图像到文件
    plt.savefig(output_image_path)
    plt.close()

# 示例使用
plot_k_fold_results('/data02/yxk/dev/s-mixup/results/GIN/REDDITB_ifplus_threshold3.txt', 'k_fold_results.png', k=2)