import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from modules import ISAB, PMA
from original_data_to_sets import get_dataloaders, BATCH_SIZE


DIM_INPUT = 10  # TODO: change default to exact number expected
DIM_HIDDEN = 256
N_HEADS = 4
N_ANC = 16


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input=DIM_INPUT,
        num_outputs=1,
        dim_output=2,
        num_inds=N_ANC,
        dim_hidden=DIM_HIDDEN,
        num_heads=N_HEADS,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_data_path", type=str)
    parser.add_argument("--background_data_path", type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dim_input", type=int, default=DIM_INPUT)
    parser.add_argument("--dim_hidden", type=int, default=DIM_HIDDEN)
    parser.add_argument("--n_heads", type=int, default=N_HEADS)
    parser.add_argument("--n_anc", type=int, default=N_ANC)
    parser.add_argument("--train_epochs", type=int, default=2000)
    args = parser.parse_args()
    args.exp_name = f"d{args.dim_input}_h{args.n_heads}_i{args.n_anc}_lr{args.learning_rate}_bs{args.batch_size}"
    log_dir = "result/" + args.exp_name
    model_path = log_dir + "/model"
    writer = SummaryWriter(log_dir)

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args.signal_data_path,
                                                                        args.background_data_path,
                                                                        batch_size=args.batch_size)

    model = SetTransformer(dim_input=args.dim_input, dim_hidden=args.dim_hidden,
                           num_heads=args.n_heads, num_inds=args.n_anc)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model = nn.DataParallel(model)
    model = model.cuda()

    ### Train ###
    for epoch in range(args.train_epochs):
        print(f"Epoch: {epoch+1}/{args.train_epochs}", end='')
        model.train()
        losses, total, correct = [], 0, 0
        for x, y in train_dataloader:
            x = x.float().cuda()
            y = y.long().cuda()
            preds = model(x)
            preds = preds.reshape(1, len(preds))
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            total += y.shape[0]
            correct += (preds.argmax(dim=1) == y).sum().item()

        avg_loss, avg_acc = np.mean(losses), correct / total
        writer.add_scalar("train_loss", avg_loss)
        writer.add_scalar("train_acc", avg_acc)
        print(f"Epoch {epoch+1}/{args.train_epochs}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")

        ### Test
        model.eval()
        losses, total, correct = [], 0, 0
        for x, y in test_dataloader:
            x = torch.stack(x).double().cuda()
            y = torch.DoubleTensor(y).cuda()
            preds = model(x)
            loss = criterion(preds, y)

            losses.append(loss.item())
            total += y.shape[0]
            correct += (preds.argmax(dim=1) == y).sum().item()
        avg_loss, avg_acc = np.mean(losses), correct / total
        writer.add_scalar("test_loss", avg_loss)
        writer.add_scalar("test_acc", avg_acc)
        print(f"Result: test loss {avg_loss:.3f} test acc {avg_acc:.3f}")
