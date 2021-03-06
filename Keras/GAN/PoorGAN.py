import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

img_rows = 28
img_cols = 28
channels = 1

# dimension
img_shape = (img_rows,img_cols, channels)

# 生成器の入力として使われるノイズベクトルの次元
z_dim = 100

# *----- 生成器の実装 -----*
def BuildGenerator(img_shape, z_dim):
    model = Sequential()
    # 全結合層
    model.add(Dense(128,input_dim=z_dim))
    # activation -> leaky ReLU
    model.add(LeakyReLU(alpha=0.01))
    # 出力層(activation -> tanh)
    model.add(Dense(28*28*1,activation='tanh'))
    # reshape
    model.add(Reshape(img_shape))
    
    return model

# *----- 識別器の実装 -----*
def BuildDiscriminator(img_shape):
    model = Sequential()

    # 入力画像を一列に並べる
    model.add(Flatten(input_shape=img_shape))
    # 全結合層
    model.add(Dense(128))
    # activation -> leaky ReLU
    model.add(LeakyReLU(alpha=0.01))
    # 出力(activation -> sigmoid)
    model.add(Dense(1,activation='sigmoid'))

    return model

def BuildGAN(generator, discriminator):
    model = Sequential()

    model.add(generator)
    model.add(discriminator)

    return model

# 識別器の構築とコンパイル
discriminator = BuildDiscriminator(img_shape)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

# 生成器の構築
generator = BuildGenerator(img_shape, z_dim)

#生成器の構築中は識別機のパラメータを固定
discriminator.trainable = False

# 生成器の訓練のため、識別機は固定し、
# GANモデルの構築とコンパイルを行っていく
gan = BuildGAN(generator,discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoints = []

# 画像の表示
def SampleImages(generator, image_grid_rows=4, image_grid_columns=4):
    # random noise のサンプリング
    z = np.random.normal(0,1,(image_grid_rows*image_grid_columns,z_dim))

    # random noise を使って画像を生成
    gen_imgs = generator.predict(z)

    # [0,1]の範囲にscale
    gen_imgs = 0.5* gen_imgs + 0.5

    # gridに並べる
    fig, axs = plt.subplot(
        image_grid_rows,
        image_grid_columns,
        figsize=(4,4),
        sharey=True,
        sharex=True
    )

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
            axs[i, j].axis('off')
            cnt += 1


#訓練
def Train(iterations, batch_size,sample_interval):
    # MNISTをロード
    (x_train, _), (_,_) = mnist.load_data()

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
        imgs = x_train[idx]

        # 偽の画像のバッチを生成
        z = np.random.normal(0,1,(batch_size,100))
        gen_imgs = generator.predict(z)

        # 訓練
        d_loss_real = discriminator.train_on_batch(imgs,real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real,d_loss_fake)

        # *----- 生成器の訓練 -----*
        # 偽の画像のバッチを生成する
        z = np.random.normal(0,1,(batch_size,100))
        gen_imgs = generator.predict(z)

        # 訓練
        g_loss = gan.train_on_batch(z,real)

        if(iteration + 1) % sample_interval == 0:
            #訓練終了後の図示の為にセーブ
            losses.append((d_loss,g_loss))
            accuracies.append(100.0*accuracy)
            iteration_checkpoints.append(iteration+1)

            #進捗を出力
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration+1,d_loss, 100.0*accuracy, g_loss))

            # 生成された画像のサンプルを出力
            sample_images(generator)

# ハイパーパラメータを設定
iterations = 20000
batch_size = 128
sample_interval = 1000

#訓練
Train(iterations,batch_size,sample_interval)