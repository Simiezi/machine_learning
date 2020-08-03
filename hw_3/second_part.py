import numpy as np
import matplotlib.pyplot as plt

# 0 == football_player
# 1 == basketball_player

football_players = np.random.randn(500) * 20 + 160
basketball_players = np.random.randn(500) * 10 + 190
foot_gt = np.zeros(len(football_players))
basket_gt = np.ones(len(basketball_players))
data_foot = np.concatenate((football_players[:, np.newaxis], foot_gt[:, np.newaxis]), axis = 1)
data_basket = np.concatenate((basketball_players[:, np.newaxis], basket_gt[:, np.newaxis]), axis = 1)
eps = 1e-10

def random_classification(height):
    return np.random.randint(2, size=1).item()


def det_classification(height, threshold):
    if height < threshold:
        return 0
    else:
        return 1


def classification(data, random, threshold=None):
    result = []
    if random:
        for elem in range(len(data)):
            result.append(random_classification(elem))
    else:
        for elem in range(len(data)):
            result.append(det_classification(elem, threshold))
    return np.array(result)


def metrics(gt, pred):
    results = [0, 0, 0, 0]
    for i in range(gt.shape[0]):
        if gt[i, 1] == 0 and pred[i, 1] == 0:
            results[0] += 1
        if gt[i, 1] == 1 and pred[i, 1] == 1:
            results[1] += 1
        if gt[i, 1] == 1 and pred[i, 1] == 0:
            results[2] += 1
        if gt[i, 1] == 0 and pred[i, 1] == 1:
            results[3] += 1
    precision = results[0] / (results[0] + results[2] + eps)
    recall = results[0] / (results[0] + results[3] + eps)
    acc = (results[0] + results[1]) / (gt.shape[0])
    results.append(precision)
    results.append(recall)
    results.append(acc * 100)
    return results


pred_foot_r = classification(football_players, True)
pred_basket_r = classification(basketball_players, True)
pred_foot_d = classification(football_players, False, 170)
pred_basket_d = classification(basketball_players, False, 170)
pred_foot_random = np.concatenate((football_players[:, np.newaxis], pred_foot_r[:, np.newaxis]), axis = 1)
pred_basket_random = np.concatenate((basketball_players[:, np.newaxis], pred_basket_r[:, np.newaxis]), axis = 1)
pred_foot_det = np.concatenate((football_players[:, np.newaxis], pred_foot_d[:, np.newaxis]), axis = 1)
pred_basket_det = np.concatenate((basketball_players[:, np.newaxis], pred_basket_d[:, np.newaxis]), axis = 1)


pred_r = np.concatenate((pred_foot_random, pred_basket_random))
pred_d = np.concatenate((pred_foot_det, pred_basket_det))
gt = np.concatenate((data_foot, data_basket))



# format tp tn fp fn prescision recall acc
r_metrics = metrics(gt, pred_r)
d_metrics = metrics(gt, pred_d)

print(f'Random metrics: TP={r_metrics[0]}, TN={r_metrics[1]}, FP={r_metrics[2]}, FN={r_metrics[3]}, Precision={r_metrics[4]:.2f}, Recall={r_metrics[5]:.2f}, Accuracy={r_metrics[6]:.2f}%')
print(f'Det metrics: TP={d_metrics[0]}, TN={d_metrics[1]}, FP={d_metrics[2]}, FN={d_metrics[3]}, Precision={d_metrics[4]:.2f}, Recall={d_metrics[5]:.2f}, Accuracy={d_metrics[6]:.2f}%')


height_list = np.arange(90, 231, 10)
recalls = []
precisions = []

for i in range(len(height_list)):
    pred_foot = classification(football_players, False, height_list[i])
    pred_basket = classification(basketball_players, False, height_list[i])
    pred_foot_full = np.concatenate((football_players[:, np.newaxis], pred_foot[:, np.newaxis]), axis = 1)
    pred_basket_full = np.concatenate((basketball_players[:, np.newaxis], pred_basket[:, np.newaxis]), axis = 1)
    prr = np.concatenate((pred_foot_full, pred_basket_full))
    metr = metrics(gt, prr)
    precisions.append(metr[4])
    recalls.append(metr[5])
print(recalls)
print(precisions)
plt.plot(precisions, recalls)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()