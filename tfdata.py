import os

import tensorflow as tf

dir = 'flower_photos/'
pastas = []

for arquivo in os.walk(dir):
    pastas.append(arquivo[0])

print(pastas)

# step 1
filenames = tf.constant(['im_01.jpg', 'im_02.jpg', 'im_03.jpg', 'im_04.jpg'])
labels = tf.constant([0, 1, 0, 1])

# step 2: create a dataset returning slices of `filenames`
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# step 3: parse every image in the dataset using `map`


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label


dataset = dataset.map(_parse_function)
dataset = dataset.batch(2)

# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()


pastas = os.listdir()
total = 0
arq = open("lista_de_arquivos.txt", "w", encoding="utf-8")


def listar_pasta(pasta):
    tot = 0
    subpastas = list()
    if os.path.isdir(pasta):
        items = os.listdir(pasta)
        print("\n\nARQUIVOS NA PASTA '" +
              str(pasta).upper() + "' :", end='\n\n')
        arq.write("ARQUIVOS NA PASTA '" + str(pasta).upper() + "': \n")
        for item in items:
            novo_item = os.path.join(pasta, item)
            if os.path.isdir(novo_item):
                subpastas.append(novo_item)
                continue
            print(item)
            arq.write(item + "\n")
            tot += 1
        for subpasta in subpastas:
            tot += listar_pasta(subpasta)
    arq.write("\n")
    return tot
