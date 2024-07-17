import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

from matplotlib import pyplot as plt

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")


    # ------------------
    #    Dataloader
    # ------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)])
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_set.transform = transform
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_set.transform = transform
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    visualization_train_loss, visualization_valid_loss = [], []
    visualization_train_acc, visualization_valid_acc = [], []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())
        mean_train_loss = np.mean(train_loss)
        mean_valid_loss =np.mean(val_loss)
        mean_train_acc = np.mean(train_acc)
        mean_valid_acc = np.mean(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {mean_train_loss:.3f} | train acc: {mean_train_acc:.3f} | val loss: {mean_valid_loss:.3f} | val acc: {mean_valid_acc:.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": mean_train_loss, "train_acc": mean_train_acc, "val_loss": mean_valid_loss, "val_acc": mean_valid_acc})
        
        visualization_train_loss.append(mean_train_loss)
        visualization_valid_loss.append(mean_valid_loss)
        visualization_train_acc.append(mean_train_acc)
        visualization_valid_acc.append(mean_valid_acc)

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
    
    # ------------------
    #   Visualization
    # ------------------
    graph_x = np.arange(1, args.epochs+1)
    plt.figure()
    plt.plot(graph_x, visualization_train_loss, label='train loss')
    plt.plot(graph_x, visualization_valid_loss, label='valid loss')
    plt.legend()
    plt.ylabel('model error')
    plt.xlabel('epochs')
    plt.title('loss')
    plt.savefig(os.path.join(logdir, "loss.png"))

    plt.figure()
    plt.plot(graph_x, visualization_train_acc, label='train acc')
    plt.plot(graph_x, visualization_valid_acc, label='valid acc')
    plt.legend()
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.title('loss')
    plt.savefig(os.path.join(logdir, "acc.png"))

    plt.figure()
    _, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(graph_x, visualization_train_loss, label='train loss')
    ax.plot(graph_x, visualization_valid_loss, label='valid loss')
    ax2.plot(graph_x, visualization_train_acc, label='train acc', color='limegreen')
    ax2.plot(graph_x, visualization_valid_acc, label='valid acc', color='black')
    ax.legend()
    ax2.legend()
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    ax2.set_ylabel('acc')
    ax.set_title('loss')
    plt.savefig(os.path.join(logdir, "loss_and_acc.png"))
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
