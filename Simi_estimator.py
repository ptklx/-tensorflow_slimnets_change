import os
import pandas as pd
import tensorflow as tf
from create_tf_record import *




labels_nums = 73  # 类别个数
batch_size = 100  #32
resize_height = 299  # 指定存储图片高度
resize_width = 299  # 指定存储图片宽度
#depths = 3
#data_shape = [batch_size, resize_height, resize_width, depths]

# 定义input_images为图片数据
#input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 定义input_labels为labels数据
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
#input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')

# 定义dropout的概率
#keep_prob = tf.placeholder(tf.float32,name='keep_prob')
#is_training = tf.placeholder(tf.bool, name='is_training')


chineseMJ= [
	"一万", "二万", "三万", "四万", "五万", "六万", "七万", "八万", "九万", 
    "一筒", "二筒", "三筒", "四筒", "五筒", "六筒", "七筒", "八筒", "九筒",
    "一条", "二条", "三条", "四条", "五条", "六条", "七条", "八条", "九条",
    "东风", "南风", "西风", "北风", "红中", "发财", "白板", 
    "春", "夏", "秋", "冬", "梅", "兰", "菊", "竹",
	"一万倒", "二万倒", "三万倒", "四万倒", "五万倒", "六万倒", "七万倒", "八万倒", "九万倒",
	"六筒倒", "七筒倒", 
	"一条倒", "三条倒", "七条倒", 
	"东风倒", "南风倒", "西风倒", "北风倒", "红中倒", "发财倒", 
	"春倒", "夏倒", "秋倒", "冬倒", "梅倒", "兰倒", "菊倒", "竹倒", "百搭", "百搭倒", "纯白" ]




#针对训练的喂食函数
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #每次随机调整数据顺序
    return dataset.make_one_shot_iterator().get_next()



#针对测试的喂食函数
def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    inputs=(features,labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size) 
    return dataset.make_one_shot_iterator().get_next()

def serving_input_fn():
    print("asdf")

def train(train_file, test_file,save_ck):
    #FUTURES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
    #SPECIES = ['Setosa', 'Versicolor', 'Virginica']

    # #格式化数据文件的目录地址
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # train_path=os.path.join(dir_path,'iris_training.csv')
    # test_path=os.path.join(dir_path,'iris_test.csv')

    # #载入训练数据
    # train = pd.read_csv(train_path, names=FUTURES, header=0)
    # train_x, train_y = train, train.pop('Species')

    # #载入测试数据
    # test = pd.read_csv(test_path, names=FUTURES, header=0)
    # test_x, test_y = test, test.pop('Species')

     # train数据,训练数据一般要求打乱顺序shuffle=True
    train_images, train_labels = read_records(train_file, resize_height, resize_width, type='normalization')
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=True)
    train_x, train_y = train_images, train_labels 
    train_nums=get_example_nums(train_file)
    # val数据,验证数据可以不需要打乱数据
    val_images, val_labels = read_records(test_file, resize_height, resize_width, type='normalization')
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)
    test_x, test_y = val_images, val_labels

    #设定特征值的名称
    feature_columns = []
    for key in range(train_nums):
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    #选定估算器：深层神经网络分类器  
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=labels_nums)

    #设定仅输出警告提示，可改为INFO
    tf.logging.set_verbosity(tf.logging.WARN)

    #开始训练模型！
    #batch_size=100
    #classifier.train(input_fn=lambda:train_input_fn(train_x, train_y,batch_size),steps=1000)  
    classifier.train([train_images_batch, train_labels_batch],steps=1000)  

    #评估我们训练出来的模型质量
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn([val_images_batch, val_labels_batch]))

    print(eval_result)

    classifier.export_savedmodel(save_ck,serving_input_fn)
        
    #支持100次循环对新数据进行分类预测
    for i in range(0,100):
        print('\nPlease enter features: SepalLength,SepalWidth,PetalLength,PetalWidth')
        a,b,c,d = map(float, input().split(',')) #捕获用户输入的数字
        predict_x = {
            'SepalLength': [a],
            'SepalWidth': [b],
            'PetalLength': [c],
            'PetalWidth': [d],
        }
        
        #进行预测
        predictions = classifier.predict(
                input_fn=lambda:eval_input_fn(predict_x,
                                                labels=[0],
                                                batch_size=batch_size))

        #预测结果是数组，尽管实际我们只有一个
        for pred_dict in predictions:
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            print(chineseMJ[class_id],100 * probability)




if __name__ == '__main__':

    #train_record_file='dataset/record/train299.tfrecords'
    #val_record_file='dataset/record/val299.tfrecords'
    #train_record_file='mjdataset/record/train299.tfrecords'
    #val_record_file='mjdataset/record_0/train299.tfrecords'
    #val_record_file='mjdataset/record_0/val299.tfrecords'

    train_record_file='dataset/record/train.tfrecords'
    val_record_file='dataset/record/val.tfrecords'
    
    train_log_step=100
    base_lr = 0.01  # 学习率
    max_steps = 10000  # 迭代次数
    train_param=[base_lr,max_steps]

    val_log_step=200
    snapshot=2000#保存文件间隔
    snapshot_prefix='mjmodels/estimator/model.ckpt'
    train(train_record_file,val_record_file,snapshot_prefix)
