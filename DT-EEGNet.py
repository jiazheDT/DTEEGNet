import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)
class ECAWeightModule(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        # 继承父类初始化
        super(ECAWeightModule, self).__init__()

        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
            kernel_size = kernel_size
        # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size + 1
        # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
        padding = kernel_size // 2
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()
    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        x = x.view([b, 1, c])
        # 1D卷积 [b,1,c]==>[b,1,c]
        x = self.conv(x)
        # 权值归一化
        x=x.view(b,c)
        x = self.sigmoid(x)
        return x
class MSEA(nn.Module):
    #这里由于通道数的限制，第三个和第四个卷积核的大小和分组数和论文里的公式略有不同
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 5, 5], stride=1, conv_groups=[1, 4, 4, 4]):
        super(MSEA, self).__init__()
        #print(inplans)
        self.conv_1 = nn.Conv2d(inplans//4,planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = nn.Conv2d(inplans//4, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = nn.Conv2d(inplans//4, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = nn.Conv2d(inplans//4, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.eca = ECAWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1=x[:,:4,:,:]
        x2=x[:, 4:8, :, :]
        x3=x[:,8:12,:,:]
        x4=x[:,12:16,:,:]
        x1 = self.conv_1(x1)
        x2 = self.conv_2(x2)
        x3 = self.conv_3(x3)
        x4 = self.conv_4(x4)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_eca = self.eca(x1)
        x2_eca = self.eca(x2)
        x3_eca = self.eca(x3)
        x4_eca = self.eca(x4)

        x_eca = torch.cat((x1_eca, x2_eca, x3_eca, x4_eca), dim=1)
        attention_vectors = x_eca.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_eca_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_eca_weight_fp
            else:
                out = torch.cat((out,x_eca_weight_fp ), 1)

        return out
class DTL_EEGNet(nn.Module):
    def __init__(self, n_classes=3, channel=32, freq=250, F1=8, D=2, F2=16, drop_prob=0.25):
        super().__init__()

        self.F2 = F2
        self.channel = channel
        self.n_classes = n_classes
        self.freq = freq
        self.F1 = F1
        self.D = D
        self.drop_prob = drop_prob

        self.conv1 = nn.Conv2d(1, self.F1, kernel_size=(1, self.freq // 2), bias=False,
                               padding=(0, self.freq // 2 // 2))
        self.bn1 = nn.BatchNorm2d(
            self.F1, momentum=0.01, affine=True, eps=1e-3)

        self.conv2 = Conv2dWithConstraint(self.F1, self.F1 * self.D, kernel_size=(self.channel, 1), max_norm=1,
                                          bias=False, groups=self.F1)
        self.bn2 = nn.BatchNorm2d(
            num_features=self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
        self.activation2 = nn.ELU(inplace=True)
        self.avg_pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(self.drop_prob)

        self.conv3_depth = nn.Conv2d(self.F1 * D, self.F1 * D, (1, 16), groups=self.F1 * D, padding=(0, 16 // 2),
                                     bias=False)
        self.conv3_point = nn.Conv2d(self.F1 * D, self.F2, (1, 1), bias=False)

        self.msea=MSEA(self.F2,self.F2)

        self.bn3 = nn.BatchNorm2d(
            self.F2, momentum=0.01, affine=True, eps=1e-3)
        self.activation3 = nn.ELU(inplace=True)
        self.avg_pool3 = nn.AvgPool2d((1, 8))

        self.drop3 = nn.Dropout(self.drop_prob)
        self.flatten = nn.Flatten()
        self.main_branch = nn.Sequential(
            LinearWithConstraint(336, 128),
            nn.ReLU(),
        )

        self.emotion_branch = nn.Sequential(
            LinearWithConstraint(128, 32),
            nn.ReLU(),
            LinearWithConstraint(32, 3),

        )

        self.cognitive_branch = nn.Sequential(
            LinearWithConstraint(128, 32),
            nn.ReLU(),
            LinearWithConstraint(32, 3),
        )


    def forward(self, x):
        x = x.reshape(-1, 1, 32, 675)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.activation2(x)
        x = self.avg_pool2(x)
        x = self.drop2(x)
        x = self.bn3(self.conv3_point(self.conv3_depth(x)))
        x = self.activation3(x)
        x = self.avg_pool3(x)
        x = self.drop3(x)
        x=self.msea(x)
        x = self.flatten(x)
        main_branch_out = self.main_branch(x)
        emotion_out = self.emotion_branch(main_branch_out)
        cognitive_out = self.cognitive_branch(main_branch_out)

        return emotion_out, cognitive_out
if __name__ == '__main__':
    # 示例使用
    model = DTL_EEGNet()
    input_tensor = torch.rand((1, 1, 32, 675))  # 示例输入
    emotion_output,cognition_output = model(input_tensor)
    print(emotion_output,cognition_output)