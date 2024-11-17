# Import necessary packages
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define a fully connected layers model with three inputs (frequency, flux density, duty ratio) and one output (power loss).
        self.layers = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),            
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_dataset(EV, mode):
    # Load data
    data_path = os.path.join('data_all', f'traindata_normalized_{EV}_{mode}.csv')
    DATA = pd.read_csv(data_path)
    
    soc = DATA.loc[:, 'SOCave292'].values.reshape(-1, 1)
    AC = DATA.loc[:, 'AC power'].values.reshape(-1, 1)
    T = DATA.loc[:, 'BMSmaxPackTemperature'].values.reshape(-1, 1)
    eff = DATA.loc[:, 'eff'].values.reshape(-1, 1)


    # Prepare input and output tensors
    temp_input = np.concatenate((AC, soc, T), axis=1)
    # temp_input = AC
    in_tensors = torch.from_numpy(temp_input).float()
    out_tensors = torch.from_numpy(eff).float()

    # Save dataset for future use
    np.save("dataset.fc.in.npy", in_tensors.numpy())
    np.save("dataset.fc.out.npy", out_tensors.numpy())


    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


# Config the model training
def main():

    
    # Reproducibility
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check CUDA availability
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)

    # Hyperparameters
    NUM_EPOCH = 1000
    BATCH_SIZE = 64
    DECAY_EPOCH = 100
    DECAY_RATIO = 0.5
    LR_INI = 0.004518782901024463

    EV = 'EV1'
    mode = 'AC'

    # Select GPU as default device
    device = torch.device("cuda:0")
    
    # Load and transform dataset, getting transformers for inverse transformation
    dataset= get_dataset(EV, mode)


    # Split the dataset
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)


    # # Split the dataset
    # train_size = int(0.9 * len(dataset))
    # valid_size = len(dataset) - train_size
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # # Set DataLoader parameters
    # kwargs = {'num_workers': 0, 'pin_memory': True}

    # # Create DataLoaders for training and validation
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    # # Later, to test using the entire dataset, you can create a test loader like this:
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)



    # Setup network
    net = Net().double().to(device).float()

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 

    # train_loss_list = np.zeros(NUM_EPOCH)
    # valid_loss_list = np.zeros(NUM_EPOCH)
        #define a list to store the validating loss in dimension of epoch*5
    valid_loss_list = np.zeros([NUM_EPOCH,1])

    #define a list to store the training loss in dimension of epoch*5
    train_loss_list = np.zeros([NUM_EPOCH,1])

    count_loss=0
    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        epoch_train_loss1 = 0
        epoch_train_loss2 = 0
        epoch_train_loss3 = 0
        epoch_train_loss4 = 0


        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))
        # optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i))      
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = net(inputs)  # Forward pass
            loss = criterion(outputs, labels)
            # loss1 = criterion(outputs[:,0], labels[:,0].to(device))
            # loss2 = criterion(outputs[:,1], labels[:,1].to(device))
            # loss3 = criterion(outputs[:,2], labels[:,2].to(device))
            # loss4 = criterion(outputs[:,3], labels[:,3].to(device))
            loss.backward()
            optimizer.step()

            # print(loss.item())

            epoch_train_loss += loss.item()
            # epoch_train_loss1 += loss1.item()
            # epoch_train_loss2 += loss2.item()
            # epoch_train_loss3 += loss3.item()
            # epoch_train_loss4 += loss4.item()


            count_loss+=1
        # Compute Validation Loss
        with torch.no_grad():
            epoch_valid_loss = 0
            # epoch_valid_loss1 = 0
            # epoch_valid_loss2 = 0
            # epoch_valid_loss3 = 0
            # epoch_valid_loss4 = 0

            for inputs, labels in valid_loader:
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                # loss1 = criterion(outputs[:,0], labels[:,0].to(device))
                # loss2 = criterion(outputs[:,1], labels[:,1].to(device))
                # loss3 = criterion(outputs[:,2], labels[:,2].to(device))
                # loss4 = criterion(outputs[:,3], labels[:,3].to(device))

                epoch_valid_loss += loss.item()
                # epoch_valid_loss1 += loss1.item()
                # epoch_valid_loss2 += loss2.item()
                # epoch_valid_loss3 += loss3.item()
                # epoch_valid_loss4 += loss4.item()

        #Save the training and validation loss into a list.
        # train_loss_list[epoch_i] = [epoch_train_loss / len(train_loader), epoch_train_loss1 / len(train_loader), epoch_train_loss2 / len(train_loader), epoch_train_loss3 / len(train_loader), epoch_train_loss4 / len(train_loader)]
        # valid_loss_list[epoch_i] = [epoch_valid_loss / len(valid_loader), epoch_valid_loss1 / len(valid_loader), epoch_valid_loss2 / len(valid_loader), epoch_valid_loss3 / len(valid_loader), epoch_valid_loss4 / len(valid_loader)]
        train_loss_list[epoch_i] = [epoch_train_loss / len(train_loader)]
        valid_loss_list[epoch_i] = [epoch_valid_loss / len(valid_loader)]
        if (epoch_i+1)%200 == 0:
          print(f"Epoch {epoch_i+1:2d} "
              f"Train {epoch_train_loss / len(train_loader):.10f} "
              f"Valid {epoch_valid_loss / len(valid_loader):.10f}")

    print(count_loss)    
    # Save the model parameters
    torch.save(net.state_dict(), f"Model_FNN_{EV}_{mode}.sd")
    print("Training finished! Model is saved!")
    
    np.savetxt(f'train_loss_overall_{EV}_{mode}.csv',train_loss_list,delimiter=',')
    np.savetxt(f'valid_loss_overall_{EV}_{mode}.csv',valid_loss_list,delimiter=',')

    net.eval()
    inputs_list = []
    y_meas = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = net(inputs.to(device))
            y_pred.append(outputs)
            y_meas.append(labels.to(device))
            inputs_list.append(inputs.to(device))

    # Concatenate all batches for full dataset predictions
    inputs_array = torch.cat(inputs_list, dim=0)

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    print(y_meas.shape, y_pred.shape)  # Debugging: Check shapes before mse_loss
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

    # Load normalization boundaries
    boundary_path = os.path.join('data_all',f'boundary_{EV}_{mode}.csv')
    boundaries = pd.read_csv(boundary_path)  #(soc,P,T)


    # soc= DATA.loc[:,'SOCave292'].values
    # AC = DATA.loc[:,'AC power'].values
    # T = DATA.loc[:,'BMSmaxPackTemperature'].values
    # DC = DATA.loc[:,'DC power'].values
    # eff = DATA.loc[:,'eff'].values
    print(boundaries)



    min_vals = boundaries.loc[boundaries['Column'].isin(['AC power','SOCave292','BMSmaxPackTemperature', 'eff']), 'min'].tolist()
    max_vals = boundaries.loc[boundaries['Column'].isin(['AC power','SOCave292','BMSmaxPackTemperature', 'eff']), 'max'].tolist()
    print(min_vals)
    # Function to reverse normalize data
    def reverse_normalize(data, min_val, max_val):
        return data * (max_val - min_val) + min_val


    # The remaining code to evaluate and save the results

    # Reverse normalization for inputs and outputs
    for i in range(inputs_array.shape[1]):
        inputs_array[:, i] = reverse_normalize(inputs_array[:, i], min_vals[i], max_vals[i])

    for i in range(y_meas.shape[1]):
        y_meas[:, i] = reverse_normalize(y_meas[:, i], min_vals[-1], max_vals[-1])
        # y_meas[:,i]=1/np.exp(y_meas[:,i])
        y_pred[:, i] = reverse_normalize(y_pred[:, i], min_vals[-1], max_vals[-1])
        # y_pred[:,i]=1/np.exp(y_pred[:,i])


    # # Retrieve mean and std values for specific columns
    # mean_vals = boundaries.loc[boundaries['Column'].isin(['AC power', 'SOCave292', 'BMSmaxPackTemperature', 'eff']), 'mean'].tolist()
    # std_vals = boundaries.loc[boundaries['Column'].isin(['AC power', 'SOCave292', 'BMSmaxPackTemperature', 'eff']), 'std'].tolist()

    # # Function to reverse Z-score normalization
    # def reverse_normalize(data, mean_val, std_val):
    #     return data * std_val + mean_val

    # # Reverse normalization for inputs and outputs
    # for i in range(inputs_array.shape[1]):
    #     inputs_array[:, i] = reverse_normalize(inputs_array[:, i], mean_vals[i], std_vals[i])

    # for i in range(y_meas.shape[1]):
    #     y_meas[:, i] = reverse_normalize(y_meas[:, i], mean_vals[-1], std_vals[-1])
    #     y_pred[:, i] = reverse_normalize(y_pred[:, i], mean_vals[-1], std_vals[-1])


    # for i in range(y_meas.shape[1]):
    #     y_meas[:, i] = y_meas[:, i]*100
    #     y_pred[:, i] = y_pred[:, i]*100
    # DataFrame to save
    df = pd.DataFrame(inputs_array, columns=[f'input_{i}' for i in range(inputs_array.shape[1])])
    output_data = {'y_meas_0': y_meas[:, 0], 'y_pred_0': y_pred[:, 0]}
                   # 'y_meas_1': y_meas[:, 1], 'y_pred_1': y_pred[:, 1],
                   # 'y_meas_2': y_meas[:, 2], 'y_pred_2': y_pred[:, 2]}
    df = pd.concat([df, pd.DataFrame(output_data)], axis=1)

    epsilon = 1e-8
    metrics = {'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'R2': []}

    # Calculate errors for each sample
    for i in range(1):
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

    # Assuming other parts of the script remain unchanged

    # Your DataFrame 'df' has been already created and populated with values
    # Let's generate scatter plots for each output variable

    output_columns = ['y_meas_0', 'y_pred_0']
    colors = ['red']

    for i in range(1):
        plt.scatter(df[f'y_meas_{i}'], df[f'y_pred_{i}'], color=colors[i], alpha=0.5)
        plt.title(f'Actual vs. Predicted Values for Output {i}\nR^2 Score: {df[f"R2_{i}"].iloc[0]:.4f}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.axis('equal')
        plt.plot([df[f'y_meas_{i}'].min(), df[f'y_meas_{i}'].max()], [df[f'y_meas_{i}'].min(), df[f'y_meas_{i}'].max()],
                 'k--')  # Diagonal line
        plt.savefig(f'output_scatter_plot_{EV}_{mode}.png')  # Save plot to file
        # plt.show()

    # # Evaluation
    # net.eval()
    # y_meas = []
    # y_pred = []
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         y_pred.append(net(inputs.to(device)))
    #         y_meas.append(labels.to(device))
    #
    # y_meas = torch.cat(y_meas, dim=0)
    # y_pred = torch.cat(y_pred, dim=0)
    # print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item():.10f}")
    #
    # # Convert tensors to numpy arrays for compatibility with pandas
    # y_meas = y_meas.cpu().numpy()
    # y_pred = y_pred.cpu().numpy()
    #
    #
    #
    # # yy_pred = 10**(y_pred)
    # # yy_meas = 10**(y_meas)
    # # Create a DataFrame with the measurements and predictions
    # df = pd.DataFrame({
    #     'y_meas': y_meas.flatten(),  # Ensure arrays are flattened
    #     'y_pred': y_pred.flatten()
    # })
    #
    # # Save the DataFrame to a CSV file
    # csv_file_path = 'y_meas_y_pred.csv'
    # df.to_csv(csv_file_path, index=False)
    #
    # print(f"Saved measurements and predictions to '{csv_file_path}'")
    #
    # # Relative Error
    # Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100
    # Error_re_avg = np.mean(Error_re)
    # Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    # Error_re_max = np.max(Error_re)
    # print(f"Relative Error: {Error_re_avg:.8f}")
    # print(f"RMS Error: {Error_re_rms:.8f}")
    # print(f"MAX Error: {Error_re_max:.8f}")

if __name__ == "__main__":

    main()