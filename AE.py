import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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


if __name__=='__main':
	txt_chg = np.load('Data/txt_chg_AE.npz')
	tensor_tc = torch.tensor(txt_chg['txt_chg'])
	tensor_tc = tensor_tc.type(torch.float)
	tc_dataset = TensorDataset(tensor_tc,tensor_tc) # create your datset
	tc_dataloader = DataLoader(tc_dataset,batch_size=512) # create your dataloader
	
    # Model Initialization
    model = AE()

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = 1e-1,
                                weight_decay = 1e-8)

    epochs = 20
    outputs = []
    losses = []

    device = set_device()
    print(device)

    if 'AE_checkpoint.pth' in os.listdir('models/'):
        checkpoint = torch.load('models/AE_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        epoch_loss = checkpoint['loss']
        model.train()
    else:
        last_epoch = 0
        epoch_loss = 0

    print(f'epoch: {last_epoch}, training loss: {epoch_loss}')
    for epoch in range(last_epoch,epochs):
        for step,(x,y) in enumerate(tqdm(tc_dataloader)):
            x = x.to(device)
            y = y.to(device)
            model = model.to(device)

            reconstructed = model(x)
            loss = loss_function(reconstructed, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Storing the losses in a list for plotting
            losses.append(loss)
    
        epoch_loss = sum(losses)/len(losses)
        print(f'epoch: {epoch}, training loss: {epoch_loss}')
        # Save Checkpoint
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    }, 'models/AE_checkpoint.pth')



    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot(losses[-100:])