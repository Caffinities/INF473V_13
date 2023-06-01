import torch
import wandb
import hydra
from tqdm import tqdm
import data.datamodule
from torch.utils.data import DataLoader
import torchvision
import cutout
from torch.cuda import amp

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):

    logger = wandb.init(project="challenge", name=f"bs{cfg.dataset.batch_size}x{cfg.loss_counter} af{cfg.af} T{cfg.confidence} unlabel{cfg.unlabelled_total} {cfg.optim._target_}{cfg.model._target_}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    enableamp = True if torch.cuda.is_available() else False
    scaler = amp.GradScaler(enabled=enableamp)
    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    
    lambda1 = lambda epoch: torch.cos(torch.tensor(7*3.1416*epoch/16/cfg.epochs))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)

    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    combinedataset = data.datamodule.combinedDataset(datamodule.train_dataset, datamodule.unlabelled_dataset, unlabelled_total=cfg.unlabelled_total)
    combined_loader = DataLoader(combinedataset, batch_size=cfg.dataset .batch_size, num_workers=cfg.dataset.num_workers, shuffle=True)
    confidence = cfg.confidence
    n_holes = cfg.holes
    cutoutlength = cfg.cutoutlength
    cut = cutout.Cutout(n_holes, cutoutlength, device)

    #threshold function
    def unlabelweight(epoch):
        alpha = 0.0
        if epoch >cfg.T2 :
                alpha = cfg.af
        elif epoch > cfg.T1:
            alpha = (epoch - cfg.T1)/(cfg.T2 - cfg.T1 + 1e-10)*cfg.af
        return alpha

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        considered_nolabel_samples = 0
        combined_loss = 0
        loss_counter = 0
        optimizer.zero_grad()
        for i, batch in enumerate(combined_loader):

            images, labels = batch
            images_strong = cut(datamodule.strong_transform(images.to(device)))
            images = datamodule.data_augment(images.to(device))
            
            labels = labels.to(device)
            with amp.autocast(enableamp):
                preds_strong = model(images_strong)
                preds = model(images)
            with torch.no_grad() and amp.autocast(enableamp):
                pseudolabels = preds.max(1)[1]
                probabilities = torch.nn.functional.softmax(preds, dim=-1).max(-1)[0]
                nolabelsize = (labels == torch.tensor([-1]*len(labels),device=device)).sum()
                #considereddatasize = (probabilities>confidence).sum()
                considered_nolabel_samples += (probabilities>confidence).float().sum().cpu().numpy()
            with amp.autocast(enableamp):
                labelledloss = loss_fn(preds, labels).mean()
                unlabelledloss = (labels.eq(-1).float()* (probabilities>confidence).float() * loss_fn(preds_strong, pseudolabels)).mean()
                loss = labelledloss + unlabelweight(epoch)*unlabelledloss 
                scaler.scale(loss/cfg.loss_counter).backward()
                num_samples += len(labels) - nolabelsize + considered_nolabel_samples
                combined_loss += labelledloss.detach().float()*(len(labels)-nolabelsize) + unlabelledloss.detach().float()*considered_nolabel_samples
                loss_counter+=1
            if loss_counter>=cfg.loss_counter:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                loss_counter=0

        scheduler.step()

        combined_loss /= num_samples
        logger.log({"combined_loss": combined_loss.detach().cpu().numpy()})
        logger.log({"considereddatasize": considered_nolabel_samples})

        num_samples = 0
        for i, batch in enumerate(train_loader):
            images, labels = batch
            images = datamodule.data_augment(images.to(device))
            labels = labels.to(device)
            with torch.no_grad():
                preds = model(images)
                loss = torch.nn.functional.cross_entropy(preds, labels, reduction='mean', label_smoothing=0.05)
            
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)

        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        for i, batch in enumerate(val_loader):
            with torch.no_grad():
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = torch.nn.functional.cross_entropy(preds,labels,label_smoothing=0.05)
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "val_loss_epoch": epoch_loss,
                "val_acc": epoch_acc,
            }
        )
    torch.save(model.state_dict(), cfg.checkpoint_path)
    wandb.finish()


if __name__ == "__main__":
    train()
