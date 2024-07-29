import numpy as np
from sklearn.model_selection import train_test_split
from models.model.transformer import Transformer
import torch
from vit_pytorch.vit import ViT
from tqdm import tqdm
from other_models import TimeSeriesCNN
from numpy.linalg import norm
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from vit_pytorch.mobile_vit import MobileViT


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cos_sim(vec1,vec2, precision=5):
    num_dim1 = len(vec1.shape)
    num_dim2 = len(vec2.shape)

    if num_dim2 != num_dim1:
        print('Vectors must have same dimensions')
        return -1

    if num_dim1 == 1:
        return round(vec1@vec2/(norm(vec1)*norm(vec2)), precision)
    elif num_dim1 == 2:
        vals = []
        for i in range(len(vec1)):
            vals.append(round(vec1[i]@vec2[i]/(norm(vec1[i])*norm(vec2[i])), precision))
        return np.array(vals)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # print(len(x.shape))
    num_dim = len(x.shape)
    if num_dim==1:
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    elif num_dim == 2:
        vals = []
        for vec in x:
            vals.append(np.exp(vec) / np.sum(np.exp(vec), axis=0))
        return np.array(vals)

def load_time_series_transformer(num_classes=19,max_len = 5000, n_head = 2, n_layer = 1, drop_prob = 0.1, d_model = 200, ffn_hidden = 128, feature = 1, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_len = 5000  # max time series sequence length
    n_head = 2  # number of attention head
    n_layer = 1  # number of encoder layer
    drop_prob = 0.1
    d_model = 200  # number of dimension ( for positional embedding)
    ffn_hidden = 128  # size of hidden layer before classification
    feature = 1  # for univariate time series (1d), it must be adjusted for 1.

    model = Transformer(num_classes=num_classes, d_model=d_model, n_head=n_head, max_len=max_len, seq_len=1000, ffn_hidden=ffn_hidden,
                        n_layers=n_layer, drop_prob=drop_prob, details=False, device=device)
    model.name = '1d_trans'
    print(f'Loading {model.name}!')
    return model

def load_time_series_CNN(num_classes=19):
    model = TimeSeriesCNN(num_classes=num_classes)
    model.name = '1d_cnn'
    print(f'Loading {model.name}!')
    return model

def load_resnet50(num_classes=19):
    from torchvision.models import resnet50
    # Load the ResNet-50 model with weights pre-trained on ImageNet
    model = resnet50(weights=None,num_classes=num_classes)
    model.name = 'resnet50'
    print(f'Loading {model.name}!')
    return model

def load_efficientnet_b4(num_classes=19):
    from torchvision.models import efficientnet_b4
    # Load the ResNet-50 model with weights pre-trained on ImageNet
    model = efficientnet_b4(weights=None,num_classes=num_classes)
    model.name = 'efficientnet_b4'
    print(f'Loading {model.name}!')
    return model

def load_mobile_vit(num_classes=19):
    model = MobileViT(
        image_size=(256, 256),
        dims=[128, 256, 512],
        channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        num_classes=num_classes
    )
    model.name = 'mobile_vit'
    print(f'Loading {model.name}!')
    return model

def load_vit(num_classes=19):
    model = ViT(
        image_size=256,
        patch_size=16,
        num_classes=19,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    model.name = 'vit'
    print(f'Loading {model.name}!')
    return model

class WaveformDataset(Dataset):
    def __init__(self, wave_data, labels):
        self.wave_data = torch.Tensor(wave_data)  # Convert to PyTorch tensor
        self.labels = torch.Tensor(labels)  # Convert to PyTorch tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_sample = self.wave_data[idx].unsqueeze(1)
        label = self.labels[idx]

        return video_sample, label

def train_model(model, X_train, y_train, y_test, X_test):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from vit_pytorch.mobile_vit import MobileViT
    import numpy as np
    model_name = model.name
    if len(X_train.shape) > 2:
        X_train_tensor = torch.tensor(np.transpose(X_train, (0, 3, 2, 1)), dtype=torch.float32)
        X_test_tensor = torch.tensor(np.transpose(X_test, (0, 3, 2, 1)), dtype=torch.float32)
    else:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # Convert numpy arrays to PyTorch tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    batch_size = 256
    if model.name in ['1d_cnn','1d_trans']:
        dataset = WaveformDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)
        dataset = WaveformDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        # Create a DataLoader for efficient batch processing

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # model = model.bfloat16()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model = model
    # Training loop
    num_epochs = 37
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f'Training on {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)
    
    model.to(device)
    
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, unit='batch'):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if len(outputs) == 2:
                try:
                    attn = outputs[1]
                    outputs = outputs[0]
                except:
                    pass

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), f'./trainedmodels/{model_name}.pth')

    # You can now use this fine-tuned model for inference or further evaluation.
    model.eval()
    total = len(X_test)
    true_labels = []
    predicted_labels = []
    cos_sims = np.array([])
    correct = 0
    total_loss = 0.0
    for inputs, labels in tqdm(test_loader, unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        if len(outputs) == 2:
            try:
                attn = outputs[1]
                outputs = outputs[0]
            except:
                pass
        loss = criterion(outputs, labels)
        # print(len(labels),len(outputs))
        # print(labels, outputs)

        cos_sims = np.append(cos_sims, cos_sim(labels.detach().cpu().numpy(), softmax(outputs.detach().cpu().numpy())))

        # print(labels)
        _, labels = torch.max(labels.data, 1)
        _, outputs = torch.max(outputs.data, 1)

        # print(labels, outputs)
        # print('outputs',outputs)
        #
        # print('labels',labels)
        # print(len(labels), len(outputs))
        # print(labels, outputs)
        total_loss += loss.item()
        correct += (outputs == labels).float().sum()

        # Collect true and predicted labels for F1 score calculation
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(outputs.cpu().numpy())

    accuracy = correct / total

    # f1 = f1_score(true_labels, predicted_labels)
    #
    # if f1 > best_val:
    #     best_val = f1
    #     torch.save(model.state_dict(), 'checkpoint_best_val.pth')
    #     print(f'F1 Score: {f1 * 100:.2f}%, saving checkpoint')
    # else:
    #     print(f'F1 Score: {f1 * 100:.2f}%')

    print(f'Testing Accuracy of {model_name}: {accuracy * 100:.2f}%')
    print(f'Average cosine similarity of {model_name}: {np.mean(cos_sims)}')

    comm_types = ['am_comm', 'cpfsk', 'bfm', 'dsb_am', 'qam']
    comm_types_53 = ['4ask', '8ask', '8psk', '16psk', '16qam', '64qam', '2fsk', '2gfsk', 'ofdm-64', 'ofdm-72']
    radar_types = ['barker', 'lfm', 'gfsk', 'rect']
    all_labels = comm_types + comm_types_53 + radar_types
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    # Visualize the confusion matrix
    plt.figure(figsize=(20, 14))


    sns.heatmap(cm_normalized, annot=True, xticklabels=all_labels, yticklabels=all_labels)


    plt.title(f'{model_name} Classification Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{model_name}_Classification_Matrix.png', format='png', dpi=1000, bbox_inches='tight')

    plt.show()
    


X_1d = np.load('datasets/256size/combined_1d_x.npy')
y_1d = np.load('datasets/256size/combined_1d_y.npy')
print(y_1d.shape)

X_2d = np.load('datasets/256size/combined_2d_x.npy')
y_2d = np.load('datasets/256size/combined_2d_y.npy')


print('Datasets loaded!')

X_train, X_test, y_train, y_test = train_test_split(X_1d, y_1d, random_state=104, test_size=0.15,shuffle=True)
np.save('datasets/testing_combined_1d_x.npy', X_test)
np.save('datasets/testing_combined_1d_y.npy', y_test)
del X_1d, y_1d

#train_model(load_time_series_CNN(),X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test)

#train_model(load_time_series_transformer(),X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test)

X_train, X_test, y_train, y_test = train_test_split(X_2d, y_2d, random_state=104, test_size=0.15,shuffle=True)
np.save('datasets/testing_combined_2d_x.npy', X_test)
np.save('datasets/testing_combined_2d_y.npy', y_test)
del X_2d, y_2d
train_model(load_mobile_vit(), X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test)

#train_model(load_vit(), X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test)

#train_model(load_efficientnet_b4(),X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test)

#train_model(load_resnet50(),X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test)
