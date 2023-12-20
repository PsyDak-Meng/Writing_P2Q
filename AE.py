import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def set_device():
    device='cuda' if torch.cuda.device_count()>0 else 'cpu'
    return device

# Creating a PyTorch class
# 120 ==> 16 ==> 120
class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(117, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 32),
			torch.nn.ReLU()
		)
		
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# The Sigmoid activation function
		# outputs the value between 0 and 1
		# 9 ==> 784
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(32, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 117),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


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


if __name__=='__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    parser = argparse.ArgumentParser(description="Choose device")
    parser.add_argument('-n','--device', default='cuda')
    args = parser.parse_args()
    print(args)
    device = args.device
    print(device)

    txt_chg = np.load('Data/txt_chg_AE.npz')
    tensor_tc = torch.tensor(txt_chg['txt_chg'])
    tensor_tc = tensor_tc.type(torch.float).to('cpu')
    tc_dataset = TensorDataset(tensor_tc,tensor_tc) # create your datset
    tc_dataloader = DataLoader(tc_dataset,batch_size=256) # create your dataloader
    
    # Initialize Model
    model = AE()

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = 1e-1,
                                weight_decay = 1e-8)

    epochs = 20
    outputs = []

    if 'AE_checkpoint.pth' in os.listdir('models/'):
        checkpoint = torch.load('models/AE_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Send to GPU
        optimizer_to(optimizer,device)
        last_epoch = checkpoint['epoch']
        last_epoch_loss = checkpoint['loss']
        model.train()
    else:
        last_epoch = 0
        last_epoch_loss = 0

    print(f'epoch: {last_epoch}, training loss: {last_epoch_loss}')
    print(f'Before training: {torch.cuda.memory_allocated(0)}')
    model = model.to(device)
    for epoch in range(last_epoch+1,epochs):
        losses = 0
        for step,(x,y) in enumerate(tqdm(tc_dataloader)):
            x = x.to(device)
            y = y.to(device)

            reconstructed = model(x)
            loss = loss_function(reconstructed, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Storing the losses in a list for plotting
            losses += loss
    
        epoch_loss = losses/(tensor_tc.size(dim=0)/256)
        print(f'epoch: {epoch}, training loss: {epoch_loss}')
        print(f'One step: {torch.cuda.memory_allocated(0)}')

        # Save Checkpoint
        if (last_epoch_loss-epoch_loss)>0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        }, 'models/AE_checkpoint.pth')
        
        torch.cuda.empty_cache()



    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot(losses[-100:])