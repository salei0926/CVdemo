import numpy as np
import os
import glob
from sklearn.utils import shuffle
import cv2

def load_train(train_path,image_size,classes):
    images = []
    labels = []
    img_names = []
    cls = []
    print('读取训练图片')
    #classes 传入列表['dogs','cats']
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read{} files (Index:{})'.format(fields,index))
        #路径读取格式：'D:/Users/16522/PycharmProjects/untitled/training_data\\dogs\\*g'
        path = os.path.join(train_path,fields,'*g')
        # 找到路径D:/Users...../dogs或者cats 且以g结尾的文件，即dogs文件夹下所有图片路径或cats文件夹下所有图片路径
        files = glob.glob(path)
        print(files)

        for f1 in files:
            print(f1)
            #image_size为图片大小，将图片转化为统一格式
            image= cv2.imread(f1)
            #统一转换为（64，64，3），通道数为3
            image = cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            #归一化处理，将数据乘以1/255，转换为0-1之间的范围u
            image = np.multiply(image,1.0/255.0)
            images.append(image)
            label = np.zeros(len(classes))
            #猫狗二分类打标签 如【1，0】
            label[index] = 1.0 #label初始化为[0,0],当为狗时[1,0],当为猫时[0,1]
            labels.append(label)
            f1base = os.path.basename(f1)  #返回path最后文件名，这里为图片名
            img_names.append(f1base)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    return images,labels,img_names,cls

class DataSet():
    def __init__(self,images,labels,img_names,cls):
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    def images(self):
        return self._images

    def labels(self):
        return self._labels

    def img_name(self):
        return self._img_names

    def cls(self):
        return self._cls

    def num_examples(self):
        return self._num_examples

    def epochs_done(self):
        return self._epochs_done

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch >self._num_examples:
            self._epochs_done += 1 #训练全部数据多少次
            start = 0
            self._index_in_epoch = batch_size
            assert  batch_size <=self._num_examples
        end = self._index_in_epoch
        return self._images[start:end],self._labels[start:end],self._img_names[start:end],self._cls[start:end]

def read_train_sets(train_path,image_size,classes,validation_size):
    class DataSets():
        pass

    data_sets = DataSets()
    images,labels,img_names,cls = load_train(train_path,image_size,classes)
    #调用sklearn.utils的shuffle方法，打散猫狗图片
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
    #这里读入2002张猫狗图片，validation_size = 0.2,因此验证集validation_size = 400
    #images (2002,64,64,3)
    if isinstance(validation_size,float):
        validation_size = int(validation_size*images.shape[0])
        validation_images = images[:validation_size]
        validation_lables = labels[:validation_size]
        validation_img_names = img_names[:validation_size]
        validation_cls = cls[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]
        train_img_names = img_names[validation_size:]
        train_cls = cls[validation_size:]

        data_sets.train = DataSet(train_images,train_labels,train_img_names,train_cls)
        data_sets.valid = DataSet(validation_images,validation_lables,validation_img_names,validation_cls)
        return data_sets




































