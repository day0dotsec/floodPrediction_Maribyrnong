"""
Isolated PyTorch utilities to avoid Streamlit file watcher conflicts
"""

def create_lstm_model():
    """Create LSTM model with completely isolated torch imports"""
    try:
        # Import torch only in function scope to avoid module inspection
        import torch
        import torch.nn as nn
        
        class LSTMFloodPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
                super(LSTMFloodPredictor, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True, dropout=dropout)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, output_size)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                out = self.sigmoid(out)
                return out
        
        return LSTMFloodPredictor, torch, nn
    
    except ImportError:
        return None, None, None

def train_lstm_isolated(X, y):
    """Train LSTM model with isolated torch imports"""
    try:
        from sklearn.model_selection import train_test_split
        
        LSTMFloodPredictor, torch, nn = create_lstm_model()
        if LSTMFloodPredictor is None:
            return None, 0.0, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
        
        model = LSTMFloodPredictor(input_size=X_train.shape[2], hidden_size=50, 
                                  num_layers=2, output_size=1)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_predictions = (test_outputs > 0.5).float()
            accuracy = (test_predictions == y_test).float().mean()
        
        return model, accuracy.item(), X_test, y_test
    
    except Exception as e:
        return None, 0.0, None, None