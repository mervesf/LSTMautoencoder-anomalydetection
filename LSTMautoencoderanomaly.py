import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('taxi_rides.csv',usecols=['value'])
df.head()

scaler=MinMaxScaler()
data_scaled=scaler.fit_transform(df)
train_size=int(0.8*len(data_scaled))
train_data=data_scaled[:train_size]
test_data=data_scaled[train_size:]


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

input_size=train_data.shape[1]
hidden_size=32

model=AutoEncoder(input_size,hidden_size)
criterion=nn.MSELoss()
optimizier=optim.Adam(model.parameters(),lr=0.001)

num_epochs = 50
for epoch in range(0, num_epochs):
    inputs = torch.tensor(train_data, dtype=torch.float32)
    outputs = model(inputs)

    loss = criterion(outputs, inputs)
    optimizier.zero_grad()
    loss.backward()
    optimizier.step()
    print(f'Epoch[{epoch + 1}/{num_epochs}],Loss:{loss.item():.4f}')

test_inputs=torch.tensor(test_data,dtype=torch.float32)
test_outputs=model(test_inputs)

test_mse=nn.functional.mse_loss(test_outputs,test_inputs,reduction='none')
test_mse=test_mse.mean(dim=1).detach().numpy()


threshold=np.mean(test_mse)+np.std(test_mse)
anomalies=test_data[test_mse>threshold]
print(test_mse)