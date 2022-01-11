import numpy as np

from nets.facenet import facenet
from nets.facenet_training import LFWDataset
from utils.utils_metrics import evaluate


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(test_loader, model):
    labels, distances = [], []

    for batch_idx, (data_a, data_p, label) in enumerate(test_loader.generate()):
        if len(label) == 0:
            break
        out_a, out_p = model.predict(data_a), model.predict(data_p)
        dists = np.linalg.norm(out_a - out_p, axis=1)

        distances.append(dists)
        labels.append(label)

        if batch_idx % log_interval == 0:
            print('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size + len(data_a), len(test_loader.validation_images),
                100. * (batch_idx * batch_size + len(data_a)) / len(test_loader.validation_images)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances,labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    plot_roc(fpr,tpr,figure_name="./model_data/roc_test.png")

def plot_roc(fpr,tpr,figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw  = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)

if __name__ == "__main__":
    #--------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet、inception_resnetv1
    #--------------------------------------#
    backbone    = "mobilenet"
    #--------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    #--------------------------------------------------------#
    input_shape = [160, 160, 3]
    #--------------------------------------#
    #   训练好的权值文件
    #--------------------------------------#
    model_path  = "model_data/facenet_mobilenet.h5"

    #--------------------------------------#
    #   评估的批次大小和记录间隔
    #--------------------------------------#
    batch_size      = 256
    log_interval    = 1

    test_loader = LFWDataset(dir="./lfw",pairs_path="model_data/lfw_pair.txt", batch_size=batch_size, input_shape=input_shape)

    model = facenet(input_shape, backbone=backbone, mode="predict")
    model.load_weights(model_path,by_name=True)

    test(test_loader, model)
