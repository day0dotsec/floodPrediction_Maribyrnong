"""
Isolated PyTorch utilities to avoid Streamlit file watcher conflicts
"""
import sys
import os

def patch_torch_classes():
    """Patch torch.classes to prevent Streamlit file watcher conflicts"""
    try:
        import torch
        # Create a safe wrapper for torch.classes that won't cause inspection issues
        if hasattr(torch, 'classes') and hasattr(torch.classes, '__path__'):
            original_path = torch.classes.__path__

            class SafeClassesPath:
                def __init__(self, original_path):
                    self._original = original_path

                def __iter__(self):
                    # Return empty iterator to avoid _path inspection issues
                    return iter([])

                @property
                def _path(self):
                    # Return empty list instead of problematic _path
                    return []

                def __getattr__(self, name):
                    # Safely delegate other attributes
                    try:
                        return getattr(self._original, name)
                    except:
                        return None

            # Replace the problematic __path__ with our safe version
            torch.classes.__path__ = SafeClassesPath(original_path)

    except Exception as e:
        # If patching fails, continue without it
        pass

def create_lstm_model():
    """Create LSTM model with completely isolated torch imports"""
    try:
        # Import torch only in function scope to avoid module inspection
        import torch
        import torch.nn as nn

        # Apply the patch immediately after import
        patch_torch_classes()
        
        class LSTMFloodPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size=3, dropout=0.2):
                super(LSTMFloodPredictor, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True, dropout=dropout)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, output_size)  # 3 classes
                self.softmax = nn.LogSoftmax(dim=1)  # For multi-class classification
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                out = self.softmax(out)  # Apply log softmax for classification
                return out
        
        return LSTMFloodPredictor, torch, nn
    
    except ImportError:
        return None, None, None

def train_lstm_isolated(X, y):
    """Train 3-class LSTM model with isolated torch imports"""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        import numpy as np

        # Apply the patch before any torch operations
        patch_torch_classes()

        LSTMFloodPredictor, torch, nn = create_lstm_model()
        if LSTMFloodPredictor is None:
            return None, 0.0, None, None, None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42, stratify=y)
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)  # LongTensor for class indices
        y_test = torch.LongTensor(y_test)

        model = LSTMFloodPredictor(input_size=X_train.shape[2], hidden_size=50,
                                  num_layers=2, output_size=3)  # 3 classes

        criterion = nn.NLLLoss()  # Negative Log Likelihood for multi-class
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Training loop with validation tracking
        model.train()
        for epoch in range(150):  # More epochs for multi-class
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Print progress every 30 epochs
            if (epoch + 1) % 30 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_accuracy = (val_predicted == y_test).float().mean()
                    print(f"Epoch [{epoch+1}/150], Loss: {loss.item():.4f}, Val Acc: {val_accuracy:.4f}")
                model.train()
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, test_predictions = torch.max(test_outputs.data, 1)
            accuracy = (test_predictions == y_test).float().mean()

            # Convert to numpy for detailed metrics
            y_test_np = y_test.numpy()
            y_pred_np = test_predictions.numpy()

            # Get classification report with metrics
            class_names = ['Low Risk', 'Medium Risk', 'High Risk']
            class_report = classification_report(y_test_np, y_pred_np,
                                                target_names=class_names, output_dict=True)
            print("\nClassification Report:")
            print(classification_report(y_test_np, y_pred_np, target_names=class_names))

            # Extract metrics for comparison
            metrics = {
                'accuracy': accuracy.item(),
                'precision': class_report['weighted avg']['precision'],
                'recall': class_report['weighted avg']['recall'],
                'f1_score': class_report['weighted avg']['f1-score']
            }

        return model, accuracy.item(), X_test, y_test, metrics
    
    except Exception as e:
        print(f"Training error: {e}")
        return None, 0.0, None, None, None