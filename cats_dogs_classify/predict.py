import glob
import tensorflow as tf
import numpy as np
import os,cv2

image_size = 64
num_channels = 3
images = []
path = 'D:\\Users\\16522\\PycharmProjects\\untitled\\cat_dog_classify\\testing_data'
#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
direct = os.listdir(path)
for file in direct:
    path = os.path.join(path,file,'*g')
    files = glob.glob(path)
    print(files)
    for f1 in files:
        print(f1)
        image = cv2.imread(f1)
        image = cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
        images.append(image)
images = np.array(images,dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images,1.0/255.0)

for img in images:
    x_batch = img.reshape(1,image_size,image_size,num_channels)
    sess = tf.Session()
    #step1网络结构图
    saver = tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-2400.meta')
    #step2加载权重参数
    saver.restore(sess,'./dogs-cats-model/dog-cat.ckpt-2400')

    #获取默认的图
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name('y_pred:0')

    x = graph.get_tensor_by_name('x:0')
    y_true = graph.get_tensor_by_name('y_true:0')
    y_test_images = np.zeros((1,2))
    feed_dict_testing = {x:x_batch,y_true:y_test_images}
    result = sess.run(y_pred,feed_dict_testing)

    res_label = ['dog','cat']

    print(res_label[result.argmax()])

