import chainer
import chainer.functions as F
import chainer.links as L

class VGGNet16(chainer.Chain):

    def __init__(self):
        super(self.__class__, self).__init__(
            conv1_1=L.Convolution2D(3, 64, ksize=3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, ksize=3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, ksize=3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, ksize=3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, ksize=3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, ksize=3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, ksize=3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, ksize=3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, ksize=3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, ksize=3, stride=1, pad=1),

            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )

    def __call__(self, x, t, train=True):
        y = self.forward(x, train=train)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss

    def forward(self, x, train=False):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=train)
        h = F.dropout(F.relu(self.fc7(h)), train=train)
        h = self.fc8(h)
        return h
