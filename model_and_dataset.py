import torch
import torch.nn as nn
from torch.utils.data import Dataset


class time_data(Dataset):
    def __init__(self, X_data, y_data,window):
        self.X_data = torch.tensor(X_data.reshape(-1,1,window),dtype=torch.float)
        self.y_data = torch.tensor(y_data.reshape(-1,1,1),dtype=torch.float)
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, index):
        return [self.X_data[index], self.y_data[index]]
        
        
class TS_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size,layers=2,output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, layers)
        self.linear= nn.Linear(hidden_size,output_dim)
        
    def forward(self, X_data):
        middle,_ = self.lstm(X_data)
        outputs = self.linear(middle.view(-1,middle.shape[2])) 
        return outputs
        
        
def get_XY(raw_data, window):
    X_data, y_data = [],[]
    for i in range(len(raw_data)-window-1):
        X_data.append(raw_data[i:i+window])
        y_data.append(raw_data[i+window])
    return np.array(X_data), np.array(y_data)

def get_dataset(csvFile, window ,boro = "Manhattan"):
    tonnage = pd.read_csv("DSNY_Monthly_Tonnage_Data.csv")
    Man = tonnage.loc[tonnage["BOROUGH"] == boro,["MONTH","REFUSETONSCOLLECTED"]].groupby(by="MONTH").sum()
    Man = Man.loc[(Man.index >="2010 / 01"),:]
    Man["scaled"] = (Man["REFUSETONSCOLLECTED"] - min(Man["REFUSETONSCOLLECTED"])) / (max(Man["REFUSETONSCOLLECTED"]) - min(Man["REFUSETONSCOLLECTED"]))
    X, y = get_XY(Man["scaled"], window)
    X_train, y_train = X[:int(len(X)*0.7)], y[:int(len(X)*0.7)]
    X_test, y_test   = X[int(len(X)*0.7):], y[int(len(X)*0.7):]
    return (X_train,y_train,X_test,y_test)