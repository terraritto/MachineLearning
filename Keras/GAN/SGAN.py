import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K

from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Concatenate, Dense, Dropout, Flatten, Input, Lambda, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

img_rows = 28
img_cols = 28
channels = 1

# dimension
img_shape = (img_rows,img_cols, channels)

# 生成器の入力として使われるノイズベクトルの次元
z_dim = 100

# データセット内のクラスの数
num_classes = 10

class Dataset:
    def __init__(self, num_labeled):
        # 訓練に用いるラベル付き訓練データ数
        self.num_labeled = num_labeled

        # MNIST dataset のロード
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        def preprocess_imgs(x):
            # [0,255]->[-1,1]
            x = (x.astype(np.float32)-127.5)/127.5
            # 画像の次元を横幅x 縦軸x チャンネル数に拡張する
            x = np.expand_dims(x,axis=3)
            return x
        
        def preprocess_labels(y):
            return y.reshape(-1,1)
        
        # train data
        self.x_train = preprocess_imgs(self.x_train)
        self.y_train = preprocess_labels(self.y_train)

        # validation data
        self.x_test = preprocess_imgs(self.x_test)
        self.y_test = preprocess_labels(self.y_test)
    
    def batch_labeled(self, batch_size):
        # ラベル付き画像と、ラベル自体をランダムに取り出してバッチを作成
        idx = np.random.randint(0, self.num_labeled, batch_size)
        imgs = self.x_train[idx]
        labels = self.y_train[idx]
        return imgs, labels

    def batch_unlabeled(self, batch_size):
        #ラベルなし画像からランダムにバッチを作成
        idx = np.random.randint(self.num_labeled, self.x_train.shape[0],batch_size)
        imgs = self.x_train[idx]
        return imgs

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train

    def test_set(self):
        return self.x_test, self.y_test      

#ラベル付き訓練データ数
num_labeled = 100

# datasetを構築
dataset = Dataset(num_labeled)

# *----- 生成器の構築-----*
# DCGANと同じ
def BuildGenerator(z_dim):
    model = Sequential()
    # 全結合層により、7x7x256のテンソルへ変換する
    model.add(Dense(256*7*7,input_dim=z_dim))
    model.add(Reshape(7,7,256))
    # 転置畳み込み層により、7x7x256->14x14x128
    model.add(Conv2DTranspose(128,kernel_size=3,strides=2,padding='same'))

    # バッチ正規化を行い、Leaky ReLUを適用
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # 転置畳み込み層により、14x14x128->14x14x64
    model.add(Conv2DTranspose(64,kernel_size=3,strides=1,padding='same'))

    # バッチ正規化を行い、Leaky ReLUを適用
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))    

    # 転置畳み込み層により、14x14x64->28x28x1
    model.add(Conv2DTranspose(1,kernel_size=3,strides=2,padding='same'))

    # tanh関数を適用して出力
    model.add(Activation('tanh'))

    return model

# *----- 識別器networkの構築 -----*
def BuildDiscriminatorNet(img_shape):
    model = Sequential()

    #畳み込み層より28x28x1->14x14x32
    model.add(
        Conv2D(
            32,
            kernel_size=3,
            strides=2,
            input_shape=img_shape,
            padding='same'
        )
    )
    # Leaky ReLU
    model.add(LeakyReLU(alpha=0.01))

    #畳み込み層より14x14x32->7x7x64
    model.add(
        Conv2D(
            64,
            kernel_size=3,
            strides=2,
            input_shape=img_shape,
            padding='same'
        )
    )
    # Batch + Leaky ReLU
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    #畳み込み層より7x7x64->3x3x128
    model.add(
        Conv2D(
            128,
            kernel_size=3,
            strides=2,
            input_shape=img_shape,
            padding='same'
        )
    )
    # Batch + Leaky ReLU
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    # dropout
    model.add(Dropout(0.5))

    # 一列に並べる
    model.add(Flatten())

    # num_classesニューロンへの全結合層
    model.add(Dense(num_classes))

    return model

# *----- 教師あり識別器の構築 -----*
def BuildDiscriminatorSupervised(discriminator_net):
    model = Sequential()

    model.add(discriminator_net)

    # 本物のクラスの中のどれに該当するかの推定確率を出力
    model.add(Activation('softmax'))

    return model

# *----- 教師なし識別器の構築 -----*
def BuildDiscriminatorUnsupervised(discriminator_net):
    model = Sequential()
    
    model.add(discriminator_net)

    def predict(x):
        # 本物のクラスにわたる確率分布を本物か偽物かの二値の確率に変換する
        prediction = 1.0 - (1.0 / (K.sum(K.exp(x),axix=-1,keepdims=True) + 1.0))
        
        return prediction
    
    #本物か偽物かを出力する
    model.add(Lambda(predict))

    return model


def BuildGAN(generator, discriminator):
    model = Sequential()

    model.add(generator)
    model.add(discriminator)

    return model

# 識別器ネットワークのコア
# 教師ありと教師なしで共有される
discriminator_net = BuildDiscriminatorNet(img_shape)

# 教師あり識別器
discriminator_supervised = BuildDiscriminatorSupervised(discriminator_net)
discriminator_supervised.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=Adam()
)

# 教師なし識別器
discriminator_unsupervised = BuildDiscriminatorUnsupervised(discriminator_net)
discriminator_unsupervised.compile(
    loss='binary_crossentropy',
    optimizer=Adam()
)

# 生成器の構築
generator = BuildGenerator(z_dim)

# 生成器の訓練中は識別器のパラメータは定数
discriminator_unsupervised.trainable = False
gan = BuildGAN(generator, discriminator_unsupervised)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

#SGANの訓練
supervised_losses = []
iteration_checkpoints = []

def Train(iterations, batch_size,sample_interval):
    # 本物の画像のラベルは全て1とする
    real = np.ones((batch_size,1))

    # 偽の画像のラベルは全て0とする
    fake = np.zeros((batch_size,1))

    for iteration in range(iterations):
        # *----- 識別器の訓練 -----*
        # ラベル付きのサンプルを得る
        imgs, labels = dataset.batch_labeled(batch_size)

        # one-hot encodingされたラベル
        labels = to_categorical(labels, num_classes=num_classes)

        # ラベルなしのサンプルを得る
        imgs_unlabeled = dataset.batch_unlabeled(batch_size)

        # 偽の画像のバッチを生成
        z = np.random.normal(0,1,(batch_size,z_dim))
        gen_imgs = generator.predict(z)

        # ラベル付きの本物のサンプルによる訓練
        d_loss_supervised, accuracy = discriminator_supervised.train_on_batch(imgs,labels)

        # ラベルなしの本物のサンプルによる訓練
        d_loss_real = discriminator_unsupervised.train_on_batch(imgs_unlabeled,real)

        # 偽のサンプルによる訓練
        d_loss_fake = discriminator_unsupervised.train_on_batch(gen_imgs, fake)
        d_loss_unsupervised = 0.5 * np.add(d_loss_real,d_loss_fake)

        # *----- 生成器の訓練 -----*
        # 偽の画像のバッチを生成する
        z = np.random.normal(0,1,(batch_size,z_dim))
        gen_imgs = generator.predict(z)

        # 訓練
        g_loss = gan.train_on_batch(z,np.ones((batch_size,1)))

        if(iteration + 1) % sample_interval == 0:
            #訓練終了後の図示の為にセーブ
            supervised_losses.append(d_loss_supervised)
            iteration_checkpoints.append(iteration+1)

            #進捗を出力
            print("%d [D loss supervised: %.4f, acc.: %.2f%%] [D loss unsupervised: %.4f] [G loss: %f]" % 
            (iteration+1,d_loss_supervised, 100.0*accuracy, d_loss_unsupervised, g_loss))

# ハイパーパラメータを設定
iterations = 20000
batch_size = 32 # ラベル付きが100個しかないので少なめ
sample_interval = 800

#訓練
Train(iterations,batch_size,sample_interval)

#精度チェック
x,y =dataset.test_set()
y = to_categorical(y,num_classes=num_classes)

#テストデータの分類度精度を計算
_, accuracy = discriminator_supervised.evaluate(x,y)
print("Test Accuracy: %.2f%%" % (100*accuracy))