import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader


class GreatNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # define network
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class FakeDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.count = 20000

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        image = torch.randn(3, 512, 512)
        return image


def multi_cuda_test():
    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('running at device %s' % default_device)
    default_type = torch.float32

    # init model
    model = GreatNetwork()
    model.to(default_device).type(default_type)
    # using multiple cuda
    model = nn.DataParallel(model)

    loss_function = nn.MSELoss()
    loss_function.to(default_device).type(default_type)

    optimizer = Adam(model.parameters(), lr=0.0001)

    batch_size = 2
    ds = DataLoader(FakeDataset(), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    position = 0
    for epoch in range(20):
        for image in ds:
            position += 1
            timestamp = time.time()
            image = image.to(default_device).type(default_type)
            optimizer.zero_grad()
            image_hat = model(image)
            loss = loss_function(image, image_hat)
            loss.backward()
            optimizer.step()
            print('TRAIN[%010d] Time: %10.4fs' % (position, time.time() - timestamp))
