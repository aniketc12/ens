from ann import Ann
import torch
import numpy as np
import torch.nn as nn

class Single_Class():
    def __init__(self, input_size, label, lr=1e-4, k=2):
        self.models = []
        self.optim = []
        self.loss = torch.nn.BCELoss()
        self.label = label
        self.k = k
        for _ in range(k):
            model = Ann(input_size)
            self.models.append(model)
            self.optim.append(torch.optim.Adam(
                model.parameters(), lr=lr,
                betas=(0.0,0.9)
            ))

    def epoch_iter(self, X, y, batch_size=32, random_bag=False):
        n, _ = X.shape
        y [y==self.label] = 1
        y [y!=self.label] = 0
        num_splits = n/batch_size
        if not num_splits % 1 == 0:
            num_splits = int(num_splits) + 1
        else:
            num_splits = int(num_splits)
        ind = np.random.choice(range(n), size=n,
                               replace=random_bag)
        Xtrain = X[ind]
        ytrain = y[ind]
        Xtrain_split = np.array_split(Xtrain, num_splits)
        ytrain_split = np.array_split(ytrain, num_splits)

        for i in range(len(Xtrain_split)):
            assert len(Xtrain_split[i]) == len(ytrain_split[i])
            currX = torch.FloatTensor(Xtrain_split[i])
            curry = torch.FloatTensor(ytrain_split[i]).unsqueeze(1)
            for j in range(self.k):
                self.optim[j].zero_grad()
                criterion = nn.BCELoss()
                y_pred = self.models[j](currX)
                loss = criterion(y_pred, curry)
                loss.backward()
                self.optim[j].step()

    def predict(self, X):
        pred = []
        with torch.no_grad():
            for i in range(self.k):
                y_pred = self.models[i](X).detach().numpy()
                pred.append(y_pred)
        return np.mean(pred, axis=0)

