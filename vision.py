import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# 混淆矩阵可视化
def confuPLT(confusion, dataset):
    # 设置标签类别
    if dataset=='MELD':
        class_names = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
    elif dataset=='IEMOCAP':
        class_names = ['happy', 'sadness', 'neutral', 'anger', 'excited', 'frustrated']
    else:
        class_names = ['happiness', 'neutral', 'anger', 'sadness', 'fear', 'surprise', 'digust']
    # 创建热图
    plt.figure(figsize=(9, 7))
    sns.set(font_scale=1.2)  # 设置字体大小
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("./fig/heatmap.png", dpi=600)