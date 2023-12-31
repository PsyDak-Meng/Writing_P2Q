import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import TensorDataset, DataLoader
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import psutil

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def load_data():
    if 'final_train_x.pt' not in os.listdir('Data/') or 'final_train_y.pt' not in os.listdir('Data/'):
        #print(psutil.virtual_memory())
        train_logs= pd.read_csv('Data/train_logs.csv')
        id = train_logs['id']
        print('id loaded...')

        tc = torch.load('Data/txt_chg_ae.pt').detach().numpy()
        print('AE loaded...',tc.shape)
    
        x = np.load('Data/x_train.npz',allow_pickle=True) 
        act = x['act']
        down = x['down']
        rest = x['rest']
        print('np loaded...',act.shape,x['down'].shape,x['rest'].shape)
        del x

        size = id.shape[0]
        x_cat = np.zeros((size,32+6+47+5))

        print('Concatenating...')
        x_cat[:,0:32] = tc
        x_cat[:,32:32+6] = act
        x_cat[:,32+6:32+6+47] = down
        x_cat[:,32+6+47:32+6+47+5] = rest

        print('Construct Dataframe...')
        x_cat = pd.DataFrame(x_cat)
        #x_cat = pd.DataFrame(x_cat.groupby(by="id", dropna=False).mean())

        train_scores = pd.read_csv('Data/train_scores.csv')
        scores = {}
        for idx, row in train_scores.iterrows():
            scores[row['id']] = row['score']
        train_scores = []
        for idx in tqdm(id,'Matching score to id...'):
            train_scores.append(scores[idx])
        train_scores = np.array(train_scores)
        train_scores = train_scores.reshape(train_scores.shape[0],1)

        del act
        del down
        del rest
        del id
        del scores

        print('y:',train_scores.shape) 
        x_cat  = np.array(x_cat)
        print('x:',x_cat.shape)

        x_cat = torch.tensor(x_cat)
        train_scores = torch.tensor(train_scores)
        torch.save(train_scores,'Data/final_train_y.pt')
        torch.save(x_cat,'Data/final_train_x.pt')
        
    
    else:
        print('Loading prepared training tensors...')
        x_cat = torch.load('Data/final_train_x.pt')
        train_scores = torch.load('Data/final_train_y.pt')


    return x_cat, train_scores



class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(input_dim,input_dim)
        self.output = nn.Linear(input_dim,1)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.matmul(queries, keys.transpose(0,1)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.matmul(attention, values)

        return weighted
    
class P2Q(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(P2Q, self).__init__()
        self.input_dim = input_dim
        self.transformer = nn.Transformer(d_model=90,nhead=6,
                                          num_encoder_layers=6,num_decoder_layers=6, 
                                          dim_feedforward=1024, dropout=0.1, activation="relu",
                                          layer_norm_eps=1e-05, 
                                          batch_first=False, norm_first=False
                                         )
        self.out = nn.Linear(90,1)


    def forward(self, src, tgt):
        x = self.transformer(src,tgt)
        x = self.out(x)
        return x
    
if __name__=='__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    parser = argparse.ArgumentParser(description="Choose device")
    parser.add_argument('-n','--device', default='cuda')
    parser.add_argument('-l','--lr', default=0.001)
    parser.add_argument('-e','--epoch', default=10)
    parser.add_argument('-d','--delete', default = 'False')
    args = parser.parse_args()
    print(args)
    device = args.device
    print(device)

    if args.delete == 'True' and 'P2Q_checkpoint.pth' in os.listdir('models/'):
        os.remove('models/P2Q_checkpoint.pth')
    
    tensor_x, tensor_y = load_data()
    tensor_x = tensor_x.type(torch.float32)
    tensor_y = tensor_y.type(torch.float32)
    print(f"input shape: {tensor_x.size()}, output shape: {tensor_y.size()}")
    dataset = TensorDataset(tensor_x, tensor_y ) # create your datset
    dataloader = DataLoader(dataset,batch_size=256) # create your dataloader
    
    # Initialize Model
    model = P2Q(input_dim=90,hidden_dim=1024)

    # Validation using MSE Loss function
    loss_function = RMSELoss

    # Using an Adam Optimizer with lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = float(args.lr),
                                weight_decay = 1e-8)

    epochs = int(args.epoch)
    outputs = []

    if 'P2Q_checkpoint.pth' in os.listdir('models/'):
        checkpoint = torch.load('models/P2Q_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Send to GPU
        optimizer_to(optimizer,device)
        last_epoch = int(checkpoint['epoch'])
        last_epoch_loss = checkpoint['loss']
        model.train()
    else:
        last_epoch = 0
        last_epoch_loss = np.inf

    print(f'epoch: {last_epoch}, training loss: {last_epoch_loss}')
    print(f'Before training: {torch.cuda.memory_allocated(0)}')
    model = model.to(device)
    for epoch in range(last_epoch+1,epochs):
        losses = 0
        for step,(x,y) in enumerate(tqdm(dataloader)):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x,x)
            loss = loss_function(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Storing the losses in a list for plotting
            losses += loss
    
        epoch_loss = losses/(tensor_y.size(dim=0)/256)
        print(f'epoch: {epoch}, training loss: {epoch_loss}')
        print(f'One step: {torch.cuda.memory_allocated(0)}')

        # Save Checkpoint
        if epoch_loss<last_epoch_loss:
            last_epoch_loss = epoch_loss
            print(f'Saving epoch {epoch}')
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        }, 'models/P2Q_checkpoint.pth')
        
        torch.cuda.empty_cache()
