import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import util
from torch.autograd import Variable
import itertools
import warnings
from PIL import ImageFilter
import numpy as np
import random
import keyboard
import sys
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# Remove warning
# warnings.filterwarnings(action='ignore')
np.set_printoptions(threshold=sys.maxsize)
# Detect NaN, Inf by autograd
torch.autograd.set_detect_anomaly(True)

class MultiTargetDnn:
    # -----------------
    #  Model information - based on Gan
    # -----------------
    # Input : Image
    # Output : Random image from noise
    # Define MSSIM loss function for binary image

    class Encoder(nn.Module):

        def reparameterization(self, mu, logvar):
            std = torch.exp(logvar / 2)
            sampledZ = Variable(torch.Tensor(np.random.normal(0, 1, (mu.size(0), self.noiseDim)))).to('cuda')
            z = sampledZ * std + mu
            return z

        def __init__(self, imgSize, noiseDim):
            super().__init__()
            # Inital size before making label Size: Batchsize x Labeldim
            # Convolution Output (Inputsize + 2*padding - kernel) / stride
            # Initial size before upsampling
            self.noiseDim = noiseDim
            self.imgShape = (3, imgSize, imgSize)
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(self.imgShape)), 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
            )

            self.mu = nn.Linear(512, self.noiseDim)
            self.logvar = nn.Linear(512, self.noiseDim)
            print("Successful loading: Encoder")

        def forward(self, img): # Labels input image
            imgFlat = img.view(img.shape[0], -1)
            x = self.model(imgFlat)
            mu = self.mu(x)
            logvar = self.logvar(x)
            z = self.reparameterization(mu, logvar)
            return z

    class Decoder(nn.Module):
        def __init__(self, imgSize, noiseDim):
            super().__init__()
            self.imgShape = (3, imgSize, imgSize)
            self.model = nn.Sequential(
                nn.Linear(noiseDim, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
                nn.Linear(512, int(np.prod(self.imgShape))),
                nn.Sigmoid(),
            )
            print("Successful loading: Decoder")
        def forward(self, z):
            imgFlat = self.model(z)
            img = imgFlat.view(imgFlat.shape[0], * self.imgShape)
            r, g, b = torch.chunk(img, 3, 1)
            return r

    class Discriminator(nn.Module):
        def __init__(self, noiseDim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(noiseDim, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        print("Successful loading: Discriminator")
        def forward(self, z):
            validity = self.model(z)
            return validity

    def sampleImage(self, nRow, batchesDone):
        """Saves a grid of generated digits"""
        # Sample noise
        z = Variable(self.Tensor(np.random.normal(0, 1, (nRow ** 2, self.noiseDim))))
        genImgs = self.decoder(z)
        save_image(genImgs.data, "binary_result/%d.png" % batchesDone, nrow=nRow, normalize=True)

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

        self.size = 224
        self.batchSize = 13
        self.epochs = 100000
        self.learningRate = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.noiseDim = 100
        self.Tensor = torch.cuda.FloatTensor
        self.preprocess = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Lambda(lambda x: x.rotate(90)),
            transforms.ToTensor(),
        ])
        self.preprocessTarget = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])
        self.encoder = self.Encoder(self.size, self.noiseDim)
        self.decoder = self.Decoder(self.size, self.noiseDim)
        self.discriminator = self.Discriminator(self.noiseDim)

        # Use binary cross-entropy loss
        self.adversarialLoss = torch.nn.BCELoss()
        self.pixelwiseLoss = torch.nn.MSELoss()
        self.trainPath = './binarys/'
        self.testPath = './binarys_test/'
        # self.validaton = []
        self.datasetTrain = util.Dataset(dataDir=self.trainPath, transform=self.preprocess, transformTarget=self.preprocessTarget, seed=seed)
        self.loaderTrain = DataLoader(self.datasetTrain, batch_size=self.batchSize, shuffle=True)
        self.datasetTest = util.Dataset(dataDir=self.testPath, transform=self.preprocess, transformTarget=self.preprocessTarget, seed=seed)
        self.loaderTest = DataLoader(self.datasetTest, batch_size=self.batchSize, shuffle=True)
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()
            self.adversarialLoss.cuda()
            self.pixelwiseLoss.cuda()

    def Train(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        self.writer = SummaryWriter()
        bestLoss = -1
        # Optimizers
        optimizerG = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.learningRate, betas=(self.b1, self.b2))
        optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.learningRate, betas=(self.b1, self.b2))
        schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizerG, T_max=100, eta_min=0)
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizerD, T_max=100, eta_min=0)

        try:  # Load Saved Model
            checkpoint = torch.load('maskganBest.model')
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            schedulerG.load_state_dict(checkpoint['lr_decay_G'])
            schedulerG.load_state_dict(checkpoint['lr_decay_D'])
            optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
            optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
            print('** Model save data is detected')
        except:
            pass

        for epoch in range(self.epochs):
            lossTotal = 0
            for batch, data in enumerate(self.loaderTrain, 1): # Total / batchsize
                if keyboard.is_pressed('F12'):
                    torch.save({'encoder': self.encoder.state_dict(),
                               'decoder': self.decoder.state_dict(),
                               'discriminator': self.discriminator.state_dict(),
                               'optimizerG_state_dict': optimizerG.state_dict(),
                               'optimizerD_state_dict': optimizerD.state_dict(),
                               'lr_decay_G': schedulerG.state_dict(),
                               'lr_decay_D': schedulerD.state_dict()},
                               'maskgan.model')
                    print('** Model is saved')
                    self.writer.flush()
                    self.writer.close()
                    exit()

                train = data['input'].to('cuda')
                target = torch.Tensor(data['target']).to('cuda')
                target = target.type(torch.float32)
                valid = Variable(self.Tensor(train.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(train.shape[0], 1).fill_(0.0), requires_grad=False)

                # -----------------
                #  Train Generator
                # -----------------
                optimizerG.zero_grad()
                encodedImgs = self.encoder(train)
                decodedImgs = self.decoder(encodedImgs)

                # Loss measures generator's ability to fool the discriminator
                gLoss = 0.001 * self.adversarialLoss(self.discriminator(encodedImgs), valid) + 0.999 * self.pixelwiseLoss(
                    decodedImgs, target
                )
                lossTotal += gLoss.item()
                gLoss.backward()
                optimizerG.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizerD.zero_grad()

                # Sample noise as discriminator ground truth
                z = Variable(self.Tensor(np.random.normal(0, 1, (train.shape[0], self.noiseDim))))

                # Measure discriminator's ability to classify real from generated samples
                realLoss = self.adversarialLoss(self.discriminator(z), valid)
                fakeLoss = self.adversarialLoss(self.discriminator(encodedImgs.detach()), fake)
                dLoss = 0.5 * (realLoss + fakeLoss)

                dLoss.backward()
                optimizerD.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, self.epochs, batch, len(self.loaderTrain), dLoss.item(), gLoss.item())
                )

                batchesDone = epoch * len(self.loaderTrain) + batch
                if batchesDone % 100 == 0:
                    self.sampleImage(nRow=10, batchesDone=batchesDone)

            schedulerG.step()
            schedulerD.step()

            self.encoder.eval()
            self.decoder.eval()

            tlossTotal = 0
            with torch.no_grad():
               for batch, data in enumerate(self.loaderTest, 1): # Total / batchsize
                    train = data['input'].to('cuda')
                    target = torch.Tensor(data['target']).to('cuda')
                    target = target.type(torch.float32)
                    encodedImgs = self.encoder(train)
                    decodedImgs = self.decoder(encodedImgs)
                    testLoss = 0.999 * self.pixelwiseLoss(
                        decodedImgs, target
                    )
                    print(
                        "Test - [Epoch %d/%d] [Batch %d/%d] [G loss: %f]" %
                        (epoch, self.epochs, batch, len(self.loaderTest), testLoss.item())
                    )
                    tlossTotal += testLoss.item()

            self.encoder.train()
            self.decoder.train()

            lossTotal = lossTotal / len(self.loaderTrain)
            tlossTotal = tlossTotal / len(self.loaderTest)

            if bestLoss == -1:  # Early Stopping
                bestLoss = tlossTotal
            else:
                if tlossTotal < bestLoss:
                    bestLoss = tlossTotal
                    torch.save({'encoder': self.encoder.state_dict(),
                                'decoder': self.decoder.state_dict(),
                                'discriminator': self.discriminator.state_dict(),
                                'optimizerG_state_dict': optimizerG.state_dict(),
                                'optimizerD_state_dict': optimizerD.state_dict(),
                                'lr_decay_G': schedulerG.state_dict(),
                                'lr_decay_D': schedulerD.state_dict()},
                               'maskganBest.model')

            if tlossTotal > bestLoss * 2:
                print('Early stopping')
                break

            self.writer.add_scalar("Loss Gan/Train", lossTotal, epoch)
            self.writer.add_scalar("Loss Gan/Test", tlossTotal, epoch)

        torch.save({'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'lr_decay_G': schedulerG.state_dict(),
                    'lr_decay_D': schedulerD.state_dict()},
                   'maskgan.model')

        self.writer.flush()
        self.writer.close()
        print('** Model is saved')

    def RunTest(self):
        checkpoint = torch.load('maskganBest.model')
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            input = os.listdir('./binary_result/test')
            for fileName in input:
                img = Image.open('./binary_result/test'+'/'+fileName)
                img = img.convert('RGB')
                img = Image.Image.resize(img, (224, 224))
                img = self.preprocess(img).to('cuda')
                img = torch.unsqueeze(img, 0)
                embedded = self.encoder(img)
                genImgs = self.decoder(embedded)
                genImgs = genImgs.squeeze()
                genImgs = genImgs.cpu()
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(size=672),
                    transforms.ToTensor()
                ])
                output = transform(genImgs).numpy()
                output = np.where(output >= 0.5, 255, 0)  # Threshhold value 0.5
                output = output.astype(np.uint8)
                output = output.squeeze()
                output = Image.fromarray(output)
                output = output.filter(ImageFilter.MaxFilter(21))
                output.show()

model = MultiTargetDnn()
model.Train()