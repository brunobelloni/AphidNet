import glob
import os
import random

import numpy as np
import tensorflow as tf
from skimage.io import imread_collection


def weight_variable(shape):
    ''' weight_variable(shape)
    - A entrada dessa função é uma lista no formato [batch,altura,largura,profundidade], na qual batch representa o número
    de imagens processadas de uma vez. Altura, largura e profundidade representam as dimensões do volume de entrada.
    - O retorno dessa função são os valores de pesos inicializados de maneira aleatória seguindo uma distribuição normal
    com desvio padrão 0.1.
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    ''' bias_variable(shape)
    - A entrada dessa função é o número de neurônios de uma camada.
    - O retorno dessa função são os valores de bias inicializados com 0.1.
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    ''' conv2d(x, W)
    - A entrada dessa função é o volume de entrada (x) e os pesos (W) da camada, ambos no formato
    [batch,altura,largura,profundidade]. Os pesos da camada são retornados na função weight_variable.
    - O retorno dessa função é o volume de saída da camada após a operação de convolução.
    - A variável strides = [1, 1, 1, 1] representa que o passo (stride) da convolução é igual a 1 em cada uma das
    dimensões.
    - A variável padding='SAME' representa que a operação de zero padding será utilizada para que o volume de saída tenha
    a mesma dimensão do volume de entrada.
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    ''' max_pool_2x2(x)
    - A entrada dessa função é o volume de entrada (x) da camada de pooling no formato [batch,altura,largura,profundidade].
    - O retorno dessa função é o volume de saída da camada após a operação de max-pooling.
    - A variável ksize = [1, 2, 2, 1] representa que o filtro utilizado na operação de pooling tem tamanho 2x2 na altura
    e largura, e tamanho 1 na dimensão de batch e profundidade.
    - A variável strides = [1, 2, 2, 1] representa que o passo (stride) da operação de pooling é igual a 2 na altura e
    largura, e 1 na dimensão de batch e profundidade.
    - A variável padding='SAME' representa que a operação de zero padding será utilizada para que o volume de saída tenha
    dimensão igual a [batch, altura/2, largura/2, profundidade] do volume de entrada.
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


'''A variável x irá armazenar as imagens de entrada da rede. Na lista com parâmetros [None,3,10000], o None é utilizado
porque não sabemos a quantidade de imagens de entrada. O 3 representa que as imagens possuem 3 canais. E o 10.000
representa a dimensão das imagens (100x100). '''
x = tf.placeholder(tf.float32, [None, 3, 10000])

'''A variável y_ representa as classes das imagens de entrada. Na lista com parâmetros [None,2], o None é utilizado
porque não sabemos a quantidade de imagens de entrada. O 2 representa a quantidade de classes que as imagens estão
divididas. '''
y_ = tf.placeholder(tf.float32, [None, 2])

'''A função tf.reshape redimensiona a variável x para o formato de entrada que o Tensorflow aceita.'''
x_image = tf.reshape(x, [-1, 100, 100, 3])

'''A variável W_conv1 irá armazenar os pesos da primeira camada convolucional, que terá 32 filtros de tamanho 5x5 e
profundidade 3. O volume de entrada dessa camada tem dimensão [batch,100,100,3]. O volume de saída terá dimensão
igual a [batch,100,100,32]'''
W_conv1 = weight_variable([5, 5, 3, 32])

'''A variável b_conv1 irá armazenar os valores de bias para os 32 filtros da primeira camada convolucional.'''
b_conv1 = bias_variable([32])

'''A função tf.nn.relu aplica a função de ativação Relu no volume de saída da primeira camada convolucional.
A variável h_conv1 irá armazenar os valores resultante da primeira camada convolucional.'''
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

'''A variável h_pool1 irá armazenar os valores resultantes após a operação de max-pool. O volume de entrada dessa
camada tem dimensão [batch,100,100,32]. O volume de saída terá dimensão igual a [batch,50,50,32]'''
h_pool1 = max_pool_2x2(h_conv1)

'''A variável W_conv2 irá armazenar os pesos da segunda camada convolucional, que terá 64 filtros de tamanho 5x5 e
profundidade 32. O volume de entrada dessa camada tem dimensão [batch,50,50,32]. O volume de saída terá dimensão
igual a [batch,50,50,64]'''
W_conv2 = weight_variable([5, 5, 32, 64])

'''A variável b_conv2 irá armazenar os valores de bias para os 64 filtros da segunda camada convolucional.'''
b_conv2 = bias_variable([64])

'''A função tf.nn.relu aplica a função de ativação Relu no volume de saída da segunda camada convolucional. A variável
h_conv2 irá armazenar os valores resultante da segunda camada convolucional.'''
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

'''A variável h_pool2 irá armazenar os valores resultantes após a operação de max-pool. O volume de entrada dessa
camada tem dimensão [batch,50,50,64]. O volume de saída terá dimensão igual a [batch,25,25,64]'''
h_pool2 = max_pool_2x2(h_conv2)

'''A variável W_fc1 irá armazenar os pesos da primeira camada totalmente conectada. O volume de entrada dessa camada
tem dimensão [batch,25,25,64]. Na lista com parâmetros [40000, 1024], o valor 40.000 é utilizado pois são
25*25*64=40.000 conexões. 1024 representa a quantidade de neurônios nessa camada.'''
W_fc1 = weight_variable([40000, 1024])

'''A variável b_fc1 irá armazenar os valores de bias para os 1024 filtros da primeira camada totalmente conectada.'''
b_fc1 = bias_variable([1024])

'''A função tf.reshape altera o formato do volume de saída da segunda camada de pooling para o formato de entrada da
primeira camada totalmente conectada.'''
h_pool2_flat = tf.reshape(h_pool2, [-1, 40000])

'''A função tf.nn.relu aplica a função de ativação Relu após a multiplicação ponto a ponto entre o volume de entrada
e os pesos da primeira camada totalmente conectada.'''
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''A variável keep_prob conterá a porcentagem de neurônios que serão ativados na aplicação do dropout durante o
treinamento.'''
keep_prob = tf.placeholder(tf.float32)

'''A função tf.nn.dropout aplica o dropout no volume resultante após a primeira camada totalmente conectada.'''
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''A variável W_fc2 conterá os pesos da segunda camada totalmente conectada. O volume de entrada dessa camada tem
1024 valores, referentes a quantidade de neurônios da camada anterior. O segundo parâmetro com valor 2 representa as
duas classes que a rede será treinada.'''
W_fc2 = weight_variable([1024, 2])

'''A variável b_fc2 conterá os valores de bias para os dois neurônios da segunda camada totalmente conectada.'''
b_fc2 = bias_variable([2])

'''A função tf.matmul realiza a multiplicação ponto a ponto entre o volume de entrada e os pesos da segunda camada
totalmente conectada. y_conv é a variável que contém a estrutura da rede.'''
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

'''A função softmax_cross_entropy_with_logits utiliza a função cross-entropy para calcular o erro entre a saída gerada
pela CNN de uma determinada entrada e a sua classe correspondente.'''
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

'''A função tf.train.AdamOptimizer atualiza os filtros e pesos da CNN utilizando o backpropagation. A variável
train_step será utilizada para realizar o treinamento da rede.'''
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

'''As duas próximas linhas são utilizadas para computar a predição da CNN e calcular a acurácia obtida.'''
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def read_images(path):
    '''A função read_images recebe o endereço da pasta que contém a base de dados e retorna dois vetores, um contendo as
    imagens e o outro contendo a classe de cada imagem.'''
    classes = glob.glob(path + '*')
    im_files = []
    size_classes = []
    for i in classes:
        name_images_per_class = glob.glob(i + '/*')
        im_files = im_files + name_images_per_class
        size_classes.append(len(name_images_per_class))
    labels = np.zeros((len(im_files), len(classes)))

    ant = 0
    for id_i, i in enumerate(size_classes):
        labels[ant:ant + i, id_i] = 1
        ant = i
    collection = imread_collection(im_files)

    data = []
    for id_i, i in enumerate(collection):
        data.append((i.reshape(3, -1)))

    return data, labels


# A variável path contém o endereço da base de imagens
path = 'flower_photos/'

# A variável data irá receber as imagens presente na pasta especificada. Já a variável labels irá receber a classe de cada uma das imagens.
data, labels = read_images(path)

# A variável batch_size representa o número de imagens que serão processadas a cada passo de treinamento.
batch_size = 50

# A variável epochs representa o número de épocas de treinamento da rede. Uma época acontece quando todas as imagens do conjunto de treinamento passam pela rede e atualizam seus valores de pesos e filtros.
epochs = 16

# A variável percent contém a porcentagem de imagens que serão utilizadas para o treinamento.
percent = 0.5

# Os códigos das próximas 5 linhas estão apenas embaralhando a ordem das imagens e dos labels.
data_size = len(data)
idx = np.arange(data_size)
random.shuffle(idx)
data = data[idx]
labels = labels[idx]

# Formando o conjunto de treinamento com a porcentagem de imagens especificado na variável percent.
train = (data[0:np.int(data_size * percent), :, :],
         labels[0:np.int(data_size * percent), :])

# Formando o conjunto de teste com as imagens que não foram utilizadas no treinamento.
test = (data[np.int(data_size * (1 - percent)):, :, :],
        labels[np.int(data_size * (1 - percent)):, :])

# A variável train_size contém o tamanho do conjunto de treinamento.
train_size = len(train[0])

# Até aqui apenas criamos as variáveis que irão realizar as operações do Tensorflow, porém é necessário criar uma sessão para que elas posam ser executadas.
sess = tf.InteractiveSession()

# É necessário inicializar todas as variáveis
tf.initialize_all_variables().run()

# Laço para repetir o processo de treinamento pelo número de épocas especificado.
for n in range(epochs):
    # Laço para dividir o conjunto de treinamento em sub conjuntos com o tamanho especificado na variável batch_size. Cada iteração desse laço representa um batch.
    for i in range(int(np.ceil(train_size / batch_size))):
        # As próximas seis linhas de código dividem o conjunto de treinamento nos batchs.
        if (i * batch_size + batch_size <= train_size):
            batch = (train[0][i * batch_size:i * batch_size + batch_size],
                     train[1][i * batch_size:i * batch_size + batch_size])
        else:
            batch = (train[0][i * batch_size:],
                     train[1][i * batch_size:])

        # Chamando a função de treinamento da rede com o valor de dropout igual a 0.5.
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        # Exibindo a acurácia obtida utilizando o conjunto de treinamento a cada 5 iterações.
        if(n % 5 == 0):
            train_accuracy = accuracy.eval(
                feed_dict={x: batch[0],
                           y_: batch[1],
                           keep_prob: 1.0})
            print("Epoca %d, acuracia do treinamento = %g" % (n,
                                                              train_accuracy))

acuracia = accuracy.eval(feed_dict={x: test[0][:],
                                    y_: test[1][:],
                                    keep_prob: 1.0})
print("Acuracia = ", acuracia)
