import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Concatenate, Dense, Embedding, Flatten, Input, Multiply, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

img_rows = 28
img_cols = 28
channels = 1

# dimension
img_shape = (img_rows,img_cols, channels)

# 生成器の入力として使われるノイズベクトルの次元
z_dim = 100

# データセット内のクラスの数
num_classes = 10

# *----- 生成器の構築-----*
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

def BuildCGANGenerator(z_dim):
    # ランダムノイズベクトル
    z = Input(shape=(z_dim,))

    # 条件ラベル: Gが生成しなければならない番号を指定
    label = Input(shape=(1,), dtype='int32')

    #ラベル埋込
    # ラベルをz_dim次元の三つベクトルに変換
    # (batch_size,1,z_dim)の3次元テンソルとなる
    label_embedding = Embedding(num_classes, z_dim, input_length=1)(label)
    # (batch_sze,z_dim)へ
    label_embedding = Flatten()(label_embedding)

    # 要素ごとの掛け算を行う
    joined_representation = Multiply()([z, label_embedding])

    generator = BuildGenerator(z_dim)

    # 与えられたラベルを持つ画像を生成する
    conditioned_img = generator(joined_representation)

    return Model([z,label], conditioned_img)

# *----- 識別器の実装 -----*
def BuildDiscriminator(img_shape):
    model = Sequential()

    #畳み込み層より28x28x2->14x14x64
    model.add(
        Conv2D(
            64,
            kernel_size=3,
            strides=2,
            input_shape=(img_shape[0],img_shape[1],img_shape[2]+1),
            padding='same'
        )
    )
    # Leaky ReLU
    model.add(LeakyReLU(alpha=0.01))

    #畳み込み層より14x14x64->7x7x64
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

    # 出力(activation -> sigmoid)
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))

    return model

def BuildCganDiscriminator(img_shape):
    # 入力画像
    img = Input(shape=img_shape)

    # 入力画像に対するラベル
    label = Input(shape=(1, ), dtype='int32')

    # ラベルの埋込
    # (batch_size, 1, 28x28x1)の3Dテンソル
    label_embedding = Embedding(
        num_classes,
        np.prod(img_shape),
        input_length=1
    )(label)

    # (batch_size,28x28x1)のテンソルに
    label_embedding = Flatten()(label_embedding)

    # ラベル埋込を入力画像と同じ形に変形する
    label_embedding = Reshape(img_shape)(label_embedding)

    #画像にラベル埋込を結合
    Concatenated = Concatenate(axis=-1)([img,label_embedding])

    discriminator = BuildDiscriminator(img_shape)

    # 画像 - ラベルの組を分類する
    classification = discriminator(Concatenated)

    return Model([img, label], classification)

def BuildCGan(generator, discriminator):
    # ランダムなノイズベクトルz
    z = Input(shape=(z_dim,))

    # 画像のラベル
    label = Input(shape=(1,))

    # ラベルに対して生成された画像
    img = generator([z,label])

    classification = discriminator([img,label])

    # 生成器->識別器と繋がる統合モデル
    # G([z, label]) = x*
    # D(x*) = 分類結果
    model = Model([z,label], classification)

    return model

# 識別器の構築とコンパイル
discriminator = BuildCganDiscriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

#生成器の訓練のための構築とコンパイル
generator = BuildCGANGenerator(z_dim)
discriminator.trainable = False

cgan = BuildCGan(generator, discriminator)
cgan.compile(loss='binary_crossentropy', optimizer=Adam())

#訓練
def Train(iterations, batch_size,sample_interval):
    # MNISTをロード
    (x_train, y_train), (_,_) = mnist.load_data()

    # [0,255]の範囲のグレースケール画素値を[-1,1]にscaling
    x_train = x_train / 127.5 - 1.0
    x_train = np.expand_dims(x_train,axis=3)

    # 本物の画像のラベルは全て1とする
    real = np.ones((batch_size,1))

    # 偽の画像のラベルは全て0とする
    fake = np.zeros((batch_size,1))

    for iteration in range(iterations):
        # *----- 識別器の訓練 -----*
        # 本物の画像をランダムに取り出したバッチを作る
        idx = np.random.randint(0,x_train.shape[0],batch_size)
        imgs, labels = x_train[idx] , y_train[idx]

        # 偽の画像のバッチを生成
        z = np.random.normal(0,1,(batch_size,z_dim))
        gen_imgs = generator.predict([z,labels])

        # 訓練
        d_loss_real = discriminator.train_on_batch([imgs, labels],real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real,d_loss_fake)

        # *----- 生成器の訓練 -----*
        # ノイズベクトルから成るbatchを生成する
        z = np.random.normal(0,1,(batch_size,z_dim))
        # ランダムなラベルを持つbatchを生成する
        labels = np.random.randint(0,num_classes, batch_size).reshape(-1,1)
        
        #生成器を訓練
        g_loss = cgan.train_on_batch([z,labels],real)

        if(iteration + 1) % sample_interval == 0:
            #進捗を出力
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration+1,d_loss[0], 100.0*d_loss[1], g_loss))
            
            #訓練終了後の図示の為にセーブ
            losses.append((d_loss[0],g_loss))
            accuracies.append(100.0*d_loss[1])
            # 生成された画像のサンプルを出力
            sample_images()

# 画像の表示
def SampleImages(image_grid_rows=2, image_grid_columns=5):
    # random noise のサンプリング
    z = np.random.normal(0,1,(image_grid_rows*image_grid_columns,z_dim))

    # 0-9の画像のラベルを得る
    labels = np.arange(0,10).reshape(-1,1)

    # random noise を使って画像を生成
    gen_imgs = generator.predict([z,labels])

    # [0,1]の範囲にscale
    gen_imgs = 0.5* gen_imgs + 0.5

    # gridに並べる
    fig, axs = plt.subplot(
        image_grid_rows,
        image_grid_columns,
        figsize=(10,4),
        sharey=True,
        sharex=True
    )

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title("Digit: %d" % labels[cnt])
            cnt += 1

# ハイパーパラメータを設定
iterations = 12000
batch_size = 32
sample_interval = 1000

#訓練
Train(iterations,batch_size,sample_interval)