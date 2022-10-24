import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_msssim
import os
import util
from torchvision.models import resnet152, resnet18, resnet34
import warnings
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
import keyboard
import sys
from torch.utils.tensorboard import SummaryWriter

# Remove warning
# warnings.filterwarnings(action='ignore')
np.set_printoptions(threshold=sys.maxsize)
# Detect NaN, Inf by autograd
torch.autograd.set_detect_anomaly(True)

class MultiTargetDnn:

    # Define MSSIM loss function for binary image
    class MSSSIM(torch.nn.Module):
        def __init__(self, windowSize=11, sizeAverage=True, valRange=1):
            super().__init__()
            self.windowSize = windowSize
            self.sizeAverage = sizeAverage
            # Max value sigmoid : 1, tanh: 2
            self.valRange = valRange

        def msssim(self, img1, img2, windowSize=11, sizeAverage=True, valRange=None, normalize=None):
            device = img1.device
            weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
            levels = weights.size()[0]
            ssims = []
            mcs = []
            for _ in range(levels):
                sim, cs = pytorch_msssim.ssim(img1, img2, window_size=windowSize, size_average=sizeAverage, full=True,
                               val_range=valRange)
                # Relu normalize (not compliant with original definition)
                if normalize == "relu":
                    ssims.append(torch.relu(sim))
                    mcs.append(torch.relu(cs))
                else:
                    ssims.append(sim)
                    mcs.append(cs)

                img1 = F.avg_pool2d(img1, (2, 2))
                img2 = F.avg_pool2d(img2, (2, 2))

            ssims = torch.stack(ssims)
            mcs = torch.stack(mcs)

            # Simple normalize (not compliant with original definition)
            # TODO: remove support for normalize == True (kept for backward support)
            if normalize == "simple" or normalize == True:
                ssims = (ssims + 1) / 2
                mcs = (mcs + 1) / 2

            pow1 = mcs ** weights
            # Remove nan if it in array
            pow2 = ssims ** weights
            # Remove nan if it in array

            # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
            output = torch.prod(pow1[:-1]) * pow2[-1]
            return output

        def forward(self, img1, img2):
            # TODO: store window between calls if possible
            return self.msssim(img1, img2, windowSize=self.windowSize, sizeAverage=self.sizeAverage, valRange=self.valRange)

    class Net(nn.Module):
        # -----------------
        #  Model information - based on Gan
        # -----------------
        # Input : Generate image from noise + input image dim
        # Output : Random image from noise

        def __init__(self, imgSize, noiseDim):
            super().__init__()
            # Inital size before making label Size: Batchsize x Labeldim
            # Convolution Output (Inputsize + 2*padding - kernel) / stride
            # Initial size before upsampling
            self.initSize = imgSize // 4
            self.l1 = nn.Sequential(
                # 128 Channel Numbers, self.initSize ** 2 Width, Height
                nn.BatchNorm1d(noiseDim),
                nn.Linear(noiseDim, 256 * self.initSize ** 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )

            self.body = nn.Sequential(
                nn.BatchNorm2d(256),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 1, 3, stride=1, padding=1),
                nn.Dropout(0.5),
                nn.Sigmoid(),
            )

            self.stam = resnet152(weights=None)
            features = self.stam.fc.in_features
            self.stam.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(features, noiseDim)
            )

            # Make label from input x using convolution
            # Output = 1 x labelDim
            # self.stam = nn.Sequential(
            #     nn.Conv2d(1, 1, 3, stride=2, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(0.25),
            #
            #     nn.BatchNorm2d(1),
            #     nn.Conv2d(1, 1, 3, stride=2, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(0.25),nn.Conv2d(1, 1, 3, stride=2, padding=1),
            #
            #     nn.BatchNorm2d(1),
            #     nn.Conv2d(1, 1, 3, stride=2, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(0.25),
            #
            #     nn.BatchNorm2d(1),
            #     nn.Conv2d(1, 1, 3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(0.25),
            #
            #     nn.BatchNorm2d(1),
            #     nn.Conv2d(1, 1, 3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(0.25),
            #
            #     nn.BatchNorm2d(1),
            #     nn.Conv2d(1, 1, 3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(0.25),
            #
            #     nn.BatchNorm2d(1),
            #     nn.Conv2d(1, 1, 3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(0.25),
            #
            #     nn.BatchNorm2d(1),
            #     nn.Conv2d(1, 1, 3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(0.25),
            #
            # )

            print("Successful loading: Net")

        def forward(self, noise): # Labels input image
            noise = self.stam(noise)
            noise = noise.view([noise.shape[0], -1])
            out = self.l1(noise)
            out = out.view(out.shape[0], 256, self.initSize, self.initSize)
            img = self.body(out)
            return img

    def __init__(self):
        seed = 17 # Random seed
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1" #
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" #
        random.seed(seed) #
        os.environ['PYTHONHASHSEED'] = str(seed) #
        np.random.seed(seed) #
        torch.manual_seed(seed) #
        torch.cuda.manual_seed(seed) #
        torch.backends.cudnn.deterministic = True #

        self.size = 112
        self.sizeTarget = 320
        self.batchSize = 13
        self.epochs = 500
        self.learningRate = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.noiseDim = 300
        self.preprocess = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])
        self.preprocessTarget = transforms.Compose([
            transforms.Resize(self.sizeTarget),
            transforms.ToTensor(),
        ])
        self.generator = self.Net(self.size, self.noiseDim)
        self.generator.to('cuda') # Use cuda
        # self.generator.apply(self.initWeight)
        self.criterion = self.MSSSIM().to('cuda')
        self.trainPath = './binarys/'
        self.testPath = './binarys_test/'
        # self.validaton = []
        self.datasetTrain = util.Dataset(dataDir=self.trainPath, transform=self.preprocess, transformTarget=self.preprocessTarget, seed=seed)
        self.loaderTrain = DataLoader(self.datasetTrain, batch_size=self.batchSize, shuffle=True)
        self.datasetTest = util.Dataset(dataDir=self.testPath, transform=self.preprocess, transformTarget=self.preprocessTarget, seed=seed)
        self.loaderTest = DataLoader(self.datasetTest, batch_size=self.batchSize, shuffle=False)

    def initWeight(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def Train(self):
        self.generator.train()
        self.writer = SummaryWriter()
        bestLoss = -1
        optimizer = optim.Adam(self.generator.parameters(), lr=self.learningRate)
        # try:  # Load Saved Model
        #     checkpoint = torch.load('best.model')
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     self.trainEpoch = checkpoint['epoch']
        #     # loss = checkpoint['loss']
        #     self.model.train()
        #     # self.model.eval() for test
        #     print('** Model save data is detected')
        #     print('* Train Epoch:', self.trainEpoch)
        # except:
        #     pass
        for epoch in range(self.epochs):
            lossTotal = 0
            idx = 1
            for batch, data in enumerate(self.loaderTrain, 1): # Total / batchsize
                if keyboard.is_pressed('F12'):
                    torch.save({'epoch': epoch,
                                'model_state_dict': self.generator.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss},
                               'Maskgan.model')
                    print('** Model is saved')
                    self.writer.flush()
                    self.writer.close()
                    exit()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer.zero_grad()
                train = data['input'].to('cuda')
                output = self.generator(train)
                target = torch.Tensor(data['target']).to('cuda')
                target = target.type(torch.float32)
                loss = self.criterion(output, target)
                print('Train Epoch:', '%d' % (epoch), 'loss = %f' % (loss.item()))
                lossTotal += loss.item()

                loss.backward()
                optimizer.step()
                idx += 1

            lossTotal = lossTotal/len(self.loaderTrain)
            self.writer.add_scalar("Loss/Train", lossTotal, epoch)

            lossTotal = 0
            idx = 1
            self.generator.eval()
            with torch.no_grad():
                for data in self.loaderTest:
                    train = data['input'].to('cuda')
                    output = self.generator(train)
                    target = torch.Tensor(data['target']).to('cuda')
                    target = target.type(torch.float32)
                    loss = self.criterion(output, target)
                    lossTotal += loss.item()
                    print('Test Epoch:', '%d' % (idx+epoch), 'loss = %f' % (loss.item()))
                    idx += 1
                lossTotal = lossTotal/len(self.loaderTest)
                self.writer.add_scalar("Loss/Test", lossTotal, epoch)

            self.generator.train()

            if bestLoss == -1:  # Early Stopping
                bestLoss = lossTotal
            else:
                if lossTotal < bestLoss:
                    bestLoss = lossTotal
                    torch.save({'epoch': epoch,
                                'model_state_dict': self.generator.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss},
                               'Maskgan_best.model')
            if lossTotal > bestLoss * 1.1:
                print('* Early Stopping')
                break

        torch.save({'epoch': epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   'Maskgan.model')
        self.writer.flush()
        self.writer.close()
        print('** Model is saved')

    def RunTest(self):
        checkpoint = torch.load('Maskgan.model')
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.generator.eval()
        with torch.no_grad():
            for iteration, data in enumerate(self.loaderTest):
                train = data['input'].to('cuda')
                output = self.generator(train)
                print(output.data)
                exit()
                # # output = output.astype(int)
                # save_image(output.data, "binary_result/%d.png"%(iteration), nrow=self.batchSize, normalize=True)

model = MultiTargetDnn()
model.Train()