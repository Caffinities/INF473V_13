from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torch
import hydra
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder


@hydra.main(config_path="configs", config_name="config", version_base=None)
def plot_confusion_matrix(cfg):

    y_pred = []
    y_label = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    #data = datamodule.val_dataloader()
    data = datamodule.train_dataloader()
    model = hydra.utils.instantiate(cfg.model).to(device)

    images = []
    labels = []
    for batch in data:
        data_images, data_labels = batch
        data_images = data_images.to(device)
        data_labels = data_labels.to(device)
        images.append(data_images)
        labels.append(data_labels)

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    labels = labels.numpy()

    predictions = model(images)
    predictions = predictions.argmax(1).numpy()

    #print(predictions[:15])
    #print(labels[:15])

    cm = confusion_matrix(labels, predictions)
    cm_df = pd.DataFrame(cm, index=[k for k in range (48)], columns=[k for k in range (48)])


    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    plt.savefig('confusion_matrix.png')


if __name__ == "__main__":
    plot_confusion_matrix()