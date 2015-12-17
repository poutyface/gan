#!/usr/bin/env python
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import optimizers
import cv2

nbatch = 10
nz = 100
nc = 3
ngf = 128
ndf = 32

seed = 42
np_rng = np.random.RandomState(seed)


class Gan(Chain):

    def __init__(self):
        super(Gan, self).__init__(
            g_l1 = L.Linear(nz, ngf*8*4*4),
            g_bn1 = L.BatchNormalization(ngf*8*4*4),
            g_bn2 = L.BatchNormalization(ngf*4),
            g_bn3 = L.BatchNormalization(ngf*2),
            g_bn4 = L.BatchNormalization(ngf),
            deconv1 = L.Deconvolution2D(ngf*8, ngf*4, ksize=5, pad=1, stride=2),
            deconv2 = L.Deconvolution2D(ngf*4, ngf*2, ksize=5, pad=1, stride=2),
            deconv3 = L.Deconvolution2D(ngf*2, ngf, ksize=5, pad=1, stride=2),
            deconv4 = L.Deconvolution2D(ngf, nc, ksize=5, pad=1, stride=2),
        )

    def make_z(self, nbatch):
        return np_rng.uniform(-1., 1., size=(nbatch, nz)).astype(np.float32)


    def generate(self, z_data):
        z = chainer.Variable(z_data)
        x = self.forward(z)
        return x.data

    def __call__(self, x):
        # h1 = F.relu(self.g_bn1(self.g_l1(x)))
        # h2 = F.reshape(h1, (h1.data.shape[0],ngf*8, 4, 4))
        # h3 = F.relu(self.g_bn2(self.deconv1(h2)))
        # h4 = F.relu(self.g_bn3(self.deconv2(h3)))
        # h5 = F.relu(self.g_bn4(self.deconv3(h4)))
        # h6 = F.relu(self.deconv4(h5))
        # h7 = F.tanh(h6)
        h1 = F.leaky_relu(self.g_l1(x))
        h2 = F.reshape(h1, (h1.data.shape[0],ngf*8, 4, 4))
        h3 = F.leaky_relu(self.deconv1(h2))
        h4 = F.leaky_relu(self.deconv2(h3))
        h5 = F.leaky_relu(self.deconv3(h4))
        h6 = F.leaky_relu(self.deconv4(h5))
        h7 = F.tanh(h6)

        print h1.data.shape
        print h2.data.shape
        print h3.data.shape
        print h4.data.shape
        print h5.data.shape
        print h6.data.shape
        return h7


class Disc(Chain):
    def __init__(self):
        super(Disc, self).__init__(
            bn1 = L.BatchNormalization(ndf*2),
            bn2 = L.BatchNormalization(ndf*4),
            bn3 = L.BatchNormalization(ndf*8),
            conv1 = L.Convolution2D(nc, ndf, ksize=5, stride=2, pad=1),
            conv2 = L.Convolution2D(ndf, ndf*2, ksize=5, stride=2, pad=1),
            conv3 = L.Convolution2D(ndf*2, ndf*4, ksize=5, stride=2, pad=1),
            conv4 = L.Convolution2D(ndf*4, ndf*8, ksize=5, stride=1, pad=1),
            l1 = L.Linear(ndf*8*7*7, 1024),
            l2 = L.Linear(1024, 1)
        )

    def __call__(self, x):
        h1 = F.leaky_relu(self.conv1(x))
        h2 = F.leaky_relu(self.conv2(h1))
        h3 = F.leaky_relu(self.conv3(h2))
        h4 = F.leaky_relu(self.conv4(h3))
        print h4.data.shape
        h5 = self.l1(h4)
        h6 = self.l2(h5)
        print h6.data
        #h7 = F.sigmoid_cross_entropy(h6, y)

        print x.data.shape
        print h1.data.shape
        print h2.data.shape
        print h3.data.shape
        print h4.data.shape
        print h5.data.shape
        print h6.data.shape
        return h6


def image_samples(nbatch):
    samples = []
    for i in range(nbatch):
        img = cv2.imread('image.jpg')
        img = cv2.resize(img, (79,79))
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        img = img.transpose(2,0,1)
        img = img[None, ...]
        samples.append(img)

    return np.concatenate(samples, axis=0)


print "---"
gen = Gan()
dis = Disc()
g_opt = optimizers.Adam(alpha=0.0002, beta1=0.5)
d_opt = optimizers.Adam(alpha=0.0002, beta1=0.5)
g_opt.setup(gen)
d_opt.setup(dis)
g_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))
d_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))

example_z = gen.make_z(nbatch)

for epoch in range(50000):
    print "epoch:", epoch

    xmb = image_samples(nbatch)

    x = gen(chainer.Variable(gen.make_z(nbatch)))
    y1 = dis(x)
    l_gen = F.sigmoid_cross_entropy(y1, chainer.Variable(np.ones((nbatch, 1), dtype=np.int32)))
    l1_dis = F.sigmoid_cross_entropy(y1, chainer.Variable(np.zeros((nbatch, 1), dtype=np.int32)))

    x2 = chainer.Variable(xmb)
    y2 = dis(x2)
    l2_dis = F.sigmoid_cross_entropy(y2, chainer.Variable(np.ones((nbatch, 1), dtype=np.int32)))
    l_dis = l1_dis + l2_dis

    print "loss gen:", l_gen.data
    print "loss dis1:", l1_dis.data
    print "loss dis2:", l2_dis.data

    gen.zerograds()
    dis.zerograds()

    margin = 0.25
    if l2_dis.data < margin:
        l_gen.backward()
        g_opt.update()

    if l1_dis.data > (1.0-margin) or l2_dis.data > margin:
        l_dis.backward()
        d_opt.update()


    img = gen(chainer.Variable(example_z)).data
    img = (img * 127.5) + 127.5
    print img.shape
    img = img.reshape(-1, nc, 79, 79)
    print img.shape
    img = img.transpose(0, 2, 3, 1)
    img = img.astype(np.uint8)
    cv2.imshow("a", img[0])
    cv2.imshow("b", img[1])
    cv2.imshow("c", img[2])
    cv2.waitKey(1)
