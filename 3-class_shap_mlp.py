from utils import *

import time

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import *

import pickle
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

seed = 1

np.random.seed(seed)
torch.manual_seed(seed)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc_layers(x)

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    # 100. * correct / len(test_loader.dataset)))


# 485
white_dataset = pd.read_csv("./dataset/raw_white.csv")
white_dataset = pd.DataFrame(white_dataset.drop('Unnamed: 0', axis=1))
white_dataset['label'] = 0

# 220117, 220425, 220502, 220530, 220606, 220613, 220620, 220704, raw_gamble_recent

gamble_dataset = pd.read_csv(f"./dataset/week_gamble/raw_gamble_recent.csv")
gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))
gamble_dataset['label'] = 1

# size_gamble_dataset = len(gamble_dataset)
# gamble_dataset['임규민'] = 0
# gamble_dataset[:int(size_gamble_dataset)]['임규민'] = 10

# 632
ad_dataset = pd.read_csv("./dataset/raw_advertisement.csv")
ad_dataset = pd.DataFrame(ad_dataset.drop('Unnamed: 0', axis=1))
ad_dataset['label'] = 2

white_train = white_dataset.sample(frac=0.8, random_state=seed)
white_test = white_dataset.drop(white_train.index)

gamble_train = gamble_dataset.sample(frac=0.8, random_state=seed)
gamble_test = gamble_dataset.drop(gamble_train.index)

ad_train = ad_dataset.sample(frac=0.8, random_state=seed)
ad_test = ad_dataset.drop(ad_train.index)

##### preprocessing #####
init_train = pd.concat([white_train, gamble_train, ad_train])
init_train.fillna(0, inplace=True)

init_test = pd.concat([white_test, gamble_test, ad_test])
init_test.fillna(0, inplace=True)

total_columns = init_train.columns

x_train = init_train.drop('label', axis=1)
y_train = init_train['label']

x_test = init_test.drop('label', axis=1)
y_test = init_test['label']

batch_size = 100
num_epochs = 30
loss_fn = nn.CrossEntropyLoss()

train_ds = TensorDataset(torch.Tensor(x_train.values),
                         torch.LongTensor(y_train.values))
train_loader = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True)

test_ds = TensorDataset(torch.Tensor(x_test.values),
                        torch.LongTensor(y_test.values))
test_loader = DataLoader(test_ds,
                         batch_size=batch_size)

##### training #####
try:
    # load model if possible
    mlp = pickle.load(open('3-class-mlp.pt','rb'))

except:
    mlp = MLP(x_train.shape[1]).to(device)

    st = time.time()
    optimizer = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, num_epochs + 1):
        train(mlp, device, train_loader, loss_fn, optimizer, epoch)
        test(mlp, device, test_loader, loss_fn)

    ed = time.time()

    # print('[*] time to train baseline:', ed-st)

    # pickle.dump(mlp, open('3-class-mlp.pt','wb'))

# evaluation
with torch.no_grad():
    x_train_ = torch.Tensor(x_train.values).to(device)
    out_train = mlp(x_train_).cpu()
    _, y_pred_train = out_train.max(1)

    x_test_ = torch.Tensor(x_test.values).to(device)
    out_test = mlp(x_test_).cpu()
    _, y_pred_test = out_test.max(1)

accuracy = accuracy_score(y_train, y_pred_train)
print("Train accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")

accuracy = accuracy_score(y_test, y_pred_test)
print("Test accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")


# print(np.where(y_test != y_pred_test))


# ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
# plt.savefig('cm_base_mlp.png')

# feature_names
feature_names = total_columns.drop('label').values

try:
    shap_values = pickle.load(open('shap_val.pt','rb'))

except:
    # explainer
    explainer = shap.DeepExplainer(mlp, torch.Tensor(x_train.values).to(device))
    shap_values = explainer.shap_values(torch.Tensor(x_test.values).to(device))

    # pickle.dump(shap_values, open('shap_val.pt','wb'))

# filtering mode
#FILTER = "by-order"         # 각 클래스별 top 100 키워드 추출
FILTER = "by-thresh"        # 각 클래스별 SHAP이 0보다 큰 키워드 추출

ORD = 100
THRESH = 0.0001

# placeholder for feature sets
feat_shap = []

n_class = 3
for cls in range(n_class):
    attr = shap_values[cls]

    # calculate mean(|SHAP values|) for each class
    avg_shap = np.abs(attr).mean(0)
    l = len(avg_shap)

    # filtering by ordering
    if FILTER == 'by-order':
        idxs = np.argpartition(avg_shap, l-ORD)[-ORD:]
        keywords = set(feature_names[idxs])

    # filtering by thresholding
    elif FILTER == 'by-thresh':
        keywords = set(feature_names[avg_shap > THRESH])

    feat_shap.append(keywords)

# keywords from shap
from functools import reduce
feat_shap_all = list(reduce(set.union, feat_shap))



# filter columns
x_train_shap = x_train[feat_shap_all]
x_test_shap = x_test[feat_shap_all]


print("랄랄랄")
print(x_test_shap.shape)
exit()

train_ds = TensorDataset(torch.Tensor(x_train_shap.values),
                         torch.LongTensor(y_train.values))
train_loader = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True)

test_ds = TensorDataset(torch.Tensor(x_test_shap.values),
                        torch.LongTensor(y_test.values))
test_loader = DataLoader(test_ds,
                         batch_size=batch_size)

##### training #####
try:
    # load model if possible
    mlp_shap = pickle.load(open('3-class-shap-mlp.pt','rb'))

except:
    mlp_shap = MLP(x_train_shap.shape[1]).to(device)

    st = time.time()
    optimizer = optim.SGD(mlp_shap.parameters(), lr=0.01, momentum=0.5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        train(mlp_shap, device, train_loader, loss_fn, optimizer, epoch)
        test(mlp_shap, device, test_loader, loss_fn)

    ed = time.time()

    # print('[*] time to train shap:', ed-st)

    #pickle.dump(mlp_shap, open('3-class-shap-mlp.pt','wb'))

# evaluation
with torch.no_grad():
    x_train_ = torch.Tensor(x_train_shap.values).to(device)
    out_train = mlp_shap(x_train_).cpu()
    _, y_pred_train = out_train.max(1)

    x_test_ = torch.Tensor(x_test_shap.values).to(device)
    out_test = mlp_shap(x_test_).cpu()
    _, y_pred_test = out_test.max(1)

accuracy = accuracy_score(y_train, y_pred_train)
print("Train accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")

accuracy = accuracy_score(y_test, y_pred_test)
print("Test accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")


print("간단 데이터")
y_test = np.array(y_test)
y_pred_test = np.array(y_pred_test)

print(len(y_test))
print(y_test)
print("---------------------------------------------------")
print(y_pred_test)

print(np.where(y_test != y_pred_test))


exit()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
# plt.savefig('cm_shap_mlp.png')