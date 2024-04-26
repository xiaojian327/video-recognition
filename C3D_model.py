import torch
import torch.nn as nn


class C3D(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)) #定义卷积conv1
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2),stride=(1, 2, 2))#定义池化层pool1

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 定义卷积conv2
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 定义池化层pool2

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 定义卷积conv3a
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 定义卷积conv3b
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 定义池化层pool3

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 定义卷积conv4a
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 定义卷积conv4b
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 定义池化层pool4

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 定义卷积conv3a
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 定义卷积conv3b
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), padding=(0, 1, 1), stride=(2, 2, 2))  # 定义池化层pool5

        self.fc6 = nn.Linear(8192, 4096)#定义线性全连接层fc6
        self.fc7 = nn.Linear(4096, 4096)#定义线性全连接层fc7
        self.fc8 = nn.Linear(4096, num_classes)#定义全连接层fc8
        self.relu = nn.ReLU()#定义激活函数ReLu
        self.dropout = nn.Dropout(p=0.5)#定义Dropout
        self.__init_weight()
        #如果存在预训练权重，则加载预训练权重中的数值，也可以不利用预训练权重
        if pretrained:
            self.__load_pretrain_weights()


    def forward(self, x):
        x = self.relu(self.conv1(x))#数据经过conv1层后经过激活函数激活
        x = self.pool1(x) #数据经过pool1层进行池化操作

        x = self.relu(self.conv2(x))#数据经过conv2层后经过激活函数激活
        x = self.pool2(x)#数据经过pool2层进行池化操作

        x = self.relu(self.conv3a(x))#数据经过conv3a层后经过激活函数激活
        x = self.relu(self.conv3b(x))#数据经过conv3b层后经过激活函数激活
        x = self.pool3(x)#数据经过pool3层进行池化操作

        x = self.relu(self.conv4a(x))#数据经过conv4a层后经过激活函数激活
        x = self.relu(self.conv4b(x))#数据巾帼conv4b层后经过激活函数激活
        x = self.pool4(x)#数据经过pool4层进行池化操作

        x = self.relu(self.conv5a(x))#数据经过conv5a层后经过激活函数激活
        x = self.relu(self.conv5b(x))#数据巾帼conv5b层后经过激活函数激活
        x = self.pool5(x)#数据经过pool5层进行池化操作

        x = x.view(-1, 8192)# 512x1x4x4-->(8192, 1) 经过pool5池化以后特征图(512, 1, 4, 4),利用vie函数将其转化为(1,8192)
        x = self.relu(self.fc6(x))#维度转化以后的数据经过fc6层,并经过激活函数
        x = self.dropout(x)#经过dropout层
        x = self.relu(self.fc7(x))#数据经过fc7层,并经过激活函数
        x = self.dropout(x)#经过dropout层

        x = self.fc8(x)#数据经过fc8层,并输出

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def __load_pretrain_weights(self):

        corresp_name ={
            #Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            #Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            #Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            #Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            #Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            #Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5b.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            #fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",#Conv4a
            #fc7
            "classifier.3.weight": "fc7.weight",
            "classifier.3.bias": "fc7.bias",


        }



        p_dict = torch.load("c3d-pretrained.pth") #加载预训练权重，预训练权重是一个字典格式，里面包含定义的网络层和对应的权重层
        # print(p_dict['features.0.weight'].shape)
        # print(p_dict['features.3.weight'].shape)
        s_dict = self.state_dict()#加载自己定义的网络层的模型名和权重层
        # print(s_dict['conv1.weight'].shape)
        # print(s_dict['conv2.weight'].shape)
        #不断循环，将预训练中corresp_name中层的权重w和b赋值给我们搭建的C3D模型
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)




if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=101).to(device)

    print(summary(net, (3, 16, 112, 112)))
