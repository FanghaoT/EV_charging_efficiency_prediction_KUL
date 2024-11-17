import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import os
from sklearn.metrics import r2_score
import torch.nn.functional as F


# Set random seed for reproducibility
manualSeed = 999
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Configurations
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
batch_size = 64
nz =1   # Size of the latent vector
num_epochs = 1000
beta1 = 0.5
network_width = 64
# Training Loop
LR_INI = 0.005  # Initial learning rate
DECAY_RATIO = 0.5  # Factor by which the learning rate is reduced
DECAY_EPOCH = 100  



EV ='EV1'
mode = 'AC'

data_path = os.path.join('data_all',f'traindata_normalized_{EV}_{mode}.csv')
DATA = pd.read_csv(data_path)

# Load and preprocess data from CSV
data = pd.read_csv(data_path)  # Replace with the actual CSV file path

print(data.columns)
# Separate input and output
x = data[["AC power","SOCave292","BMSmaxPackTemperature"]].values  # Replace with the actual target column name
y = data["eff"].values  # Replace with the actual target column name

# First, split into train (80%) and temp (20%)
train_data, val_data, train_labels, val_labels = train_test_split(
    x, y, test_size=0.2, random_state=manualSeed
)
test_data = val_data
test_labels = val_labels

# Data loader
class PrepareData(torch.utils.data.Dataset):
    def __init__(self, X, y, scale_X=True):
        if not torch.is_tensor(X):
            if scale_X:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X).float().to(device)
        else:
            self.X = X.float()
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y).float().to(device)
        else:
            self.y = y.float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset instances for each split
train_ds = PrepareData(train_data, y=train_labels, scale_X=False)
val_ds = PrepareData(val_data, y=val_labels, scale_X=False)
test_ds = PrepareData(test_data, y=test_labels, scale_X=False)

# Create data loaders for each split
train_set = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_set = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_set = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(x.shape[1] + nz, network_width, bias=True),
            nn.ReLU(),
            nn.Linear(network_width, network_width, bias=True),
            nn.ReLU(),
            nn.Linear(network_width, 1, bias=True)
        )

    def forward(self, input):
        return self.main(input)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(x.shape[1] + 1, network_width, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(network_width, network_width, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(network_width, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Initialize models and optimizer
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(),betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(),betas=(beta1, 0.999))

# Training Loop
for epoch in range(num_epochs):
    # Update learning rates with decay
    new_lr = LR_INI * (DECAY_RATIO ** (epoch // DECAY_EPOCH))
    for param_group in optimizerD.param_groups:
        param_group['lr'] = new_lr
    for param_group in optimizerG.param_groups:
        param_group['lr'] = new_lr

    epoch_train_loss_D = 0.0
    epoch_train_loss_G = 0.0
    count_loss = 0

    for i, data in enumerate(train_set, 0):
        real_ = torch.hstack((data[0].to(device), data[1].unsqueeze(1).to(device)))
        b_size = real_.size(0)
        label = torch.full((b_size,), 1., dtype=torch.float, device=device)
        
        # Train Discriminator
        netD.zero_grad()
        output = netD(real_).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, nz, device=device)
        gen_ip = torch.hstack((data[0].to(device), noise))
        fake = netG(gen_ip)
        label.fill_(0)
        fake_ = torch.hstack((data[0], fake.detach())).to(device)
        output = netD(fake_.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        label.fill_(1)
        fake_ = torch.hstack((data[0], fake)).to(device)
        output = netD(fake_).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

       # Accumulate losses for reporting
        epoch_train_loss_D += (errD_real.item() + errD_fake.item())
        epoch_train_loss_G += errG.item()
        count_loss += 1
        # Print losses every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}]",f"Discriminator Loss: {epoch_train_loss_D/len(train_ds):.10f}",f"Generator Loss: {epoch_train_loss_G/len(train_ds):.10f}")


# Evaluation on test set
netG.eval()
inputs_list = []
y_meas = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_set:
        # Generate predictions
        scale_factor = 1  # adjust this value as needed to control the noise scale
        noise = torch.randn(inputs.size(0), nz, device=device) * scale_factor
        # noise = torch.randn(inputs.size(0), nz, device=device)
        gen_input = torch.hstack((inputs.to(device), noise))
        outputs = netG(gen_input.to(device))
        
        # Store values for analysis
        y_pred.append(outputs)
        y_meas.append(labels.to(device))
        inputs_list.append(inputs.to(device))

# Concatenate all batches
inputs_array = torch.cat(inputs_list, dim=0).to(device)
y_meas = torch.cat(y_meas, dim=0).to(device)
y_pred = torch.cat(y_pred, dim=0).to(device)

# Calculate overall metrics
mse_value = F.mse_loss(y_meas, y_pred).item()
rmse_value = np.sqrt(mse_value)
mae_value = F.l1_loss(y_meas, y_pred).item()
print(f"Test MSE: {mse_value:.10f}")
print(f"Test RMSE: {rmse_value:.10f}")
print(f"Test MAE: {mae_value:.10f}")

# Convert tensors to numpy arrays for further processing
inputs_array = inputs_array.cpu().numpy()
y_meas = y_meas.cpu().numpy()
y_pred = y_pred.cpu().numpy()
# Ensure y_meas and y_pred are 2D for consistency
if y_meas.ndim == 1:
    y_meas = np.expand_dims(y_meas, axis=1)  # Add a dimension to make it 2D
if y_pred.ndim == 1:
    y_pred = np.expand_dims(y_pred, axis=1)



# Load normalization boundaries
boundary_path = os.path.join('data_all', f'boundary_{EV}_{mode}.csv')
boundaries = pd.read_csv(boundary_path)
print(boundaries)

# Extract min and max values for reverse normalization
min_vals = boundaries.loc[boundaries['Column'].isin(['AC power', 'SOCave292', 'BMSmaxPackTemperature', 'eff']), 'min'].tolist()
max_vals = boundaries.loc[boundaries['Column'].isin(['AC power', 'SOCave292', 'BMSmaxPackTemperature', 'eff']), 'max'].tolist()
print("Min values:", min_vals)
print("Max values:", max_vals)

# Function to reverse normalize data
def reverse_normalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

# Reverse normalization for inputs and outputs
for i in range(inputs_array.shape[1]):
    inputs_array[:, i] = reverse_normalize(inputs_array[:, i], min_vals[i], max_vals[i])

for i in range(y_meas.shape[1]):
    y_meas[:, i] = reverse_normalize(y_meas[:, i], min_vals[-1], max_vals[-1])
    y_pred[:, i] = reverse_normalize(y_pred[:, i], min_vals[-1], max_vals[-1])

# Prepare DataFrame to save
df = pd.DataFrame(inputs_array, columns=[f'input_{i}' for i in range(inputs_array.shape[1])])
output_data = {'y_meas_0': y_meas[:, 0], 'y_pred_0': y_pred[:, 0]}
df = pd.concat([df, pd.DataFrame(output_data)], axis=1)

# Error metrics initialization
epsilon = 1e-8
metrics = {'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'R2': []}

# Calculate errors for each output
for i in range(1):  # Adjust for multiple outputs if needed
    mae = np.abs(y_meas[:, i] - y_pred[:, i])
    mse = (y_meas[:, i] - y_pred[:, i]) ** 2
    rmse = np.sqrt(mse)
    mape = np.abs((y_meas[:, i] - y_pred[:, i]) / (y_meas[:, i] + epsilon)) * 100
    r2 = r2_score(y_meas[:, i], y_pred[:, i])

    # Store each metric in the DataFrame
    df[f'MAE_{i}'] = mae
    df[f'MSE_{i}'] = mse
    df[f'RMSE_{i}'] = rmse
    df[f'MAPE_{i}'] = mape
    df[f'R2_{i}'] = r2
    # Collect mean values in a dictionary to print later
    metrics['MAE'].append(np.mean(mae))
    metrics['MSE'].append(np.mean(mse))
    metrics['RMSE'].append(np.mean(rmse))
    metrics['MAPE'].append(np.mean(mape))
    metrics['R2'].append(r2)  # R2 is already a summary metric

# Print mean of each metric for all outputs
for metric, values in metrics.items():
    print(f"Mean {metric}: {np.mean(values):.10f}")

# Save the DataFrame to a CSV file
csv_file_path = f'full_data_y_meas_y_pred_{EV}_{mode}.csv'
df.to_csv(csv_file_path, index=False)
print(f"Saved full data with measurements and predictions to '{csv_file_path}'")

# Scatter plots for each output variable
output_columns = ['y_meas_0', 'y_pred_0']
colors = ['red']

for i in range(1):
    plt.scatter(df[f'y_meas_{i}'], df[f'y_pred_{i}'], color=colors[i], alpha=0.5)
    plt.title(f'Actual vs. Predicted Values for Output {i}\nR² Score: {df[f"R2_{i}"].iloc[0]:.4f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.axis('equal')
    plt.plot(
        [df[f'y_meas_{i}'].min(), df[f'y_meas_{i}'].max()],
        [df[f'y_meas_{i}'].min(), df[f'y_meas_{i}'].max()],
        'k--'  # Diagonal line for reference
    )
    plt.savefig(f'output_scatter_plot_{EV}_{mode}.png')  # Save plot to file
    # plt.show()
