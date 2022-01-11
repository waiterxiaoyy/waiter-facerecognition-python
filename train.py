import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from nets.facenet import facenet
from nets.facenet_training import FacenetDataset, LFWDataset, triplet_loss
from utils.callbacks import ExponentDecayScheduler, LFW_callback, LossHistory


# ------------------------------------------------#
#   计算一共有多少个人，用于利用交叉熵辅助收敛
# ------------------------------------------------#
def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()
    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


if __name__ == "__main__":
    # --------------------------------------------------------#
    #   权值文件和损失曲线所保存的路径
    # --------------------------------------------------------#
    log_dir = "logs/"
    # --------------------------------------------------------#
    #   指向根目录下的cls_train.txt，读取人脸路径与标签
    # --------------------------------------------------------#
    annotation_path = "cls_train.txt"
    # --------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    # --------------------------------------------------------#
    input_shape = [160, 160, 3]
    # --------------------------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet；inception_resnetv1
    # --------------------------------------------------------#
    backbone = "mobilenet"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的训练参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，Freeze_Train = False，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = "model_data/facenet_mobilenet.h5"
    # -------------------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # -------------------------------------------------------------------#
    Freeze_Train = True
    # -------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，1代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # -------------------------------------------------------------------#
    num_workers = 1
    # -------------------------------------------------------------------#
    #   是否开启LFW评估
    # -------------------------------------------------------------------#
    lfw_eval_flag = True
    # -------------------------------------------------------------------#
    #   LFW评估数据集的文件路径和对应的txt文件
    # -------------------------------------------------------------------#
    lfw_dir_path = "lfw"
    lfw_pairs_path = "model_data/lfw_pair.txt"

    num_classes = get_num_classes(annotation_path)
    # ---------------------------------#
    #   载入模型并加载预训练权重
    # ---------------------------------#
    model = facenet(input_shape, num_classes, backbone=backbone, mode="train")
    if model_path != '':
        # ---------------------------------#
        #   载入预训练权重
        # ---------------------------------#
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # -------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    # -------------------------------------------------------------------------------#
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ExponentDecayScheduler(decay_rate=0.94, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)
    loss_history = LossHistory(log_dir)
    # ---------------------------------#
    #   LFW估计
    # ---------------------------------#
    if lfw_eval_flag:
        lfw_callback = LFW_callback(
            LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, batch_size=32, input_shape=input_shape))

    # ---------------------------------#
    #   0.05用于验证，0.95用于训练
    # ---------------------------------#
    val_split = 0.05
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    if backbone == "mobilenet":
        freeze_layer = 81
    elif backbone == "inception_resnetv1":
        freeze_layer = 440
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))

    if Freeze_Train:
        for i in range(freeze_layer):
            model.layers[i].trainable = False

    # ---------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    # ---------------------------------------------------------#
    # ---------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ---------------------------------------------------------#
    if True:
        # ----------------------------------------------------#
        #   冻结阶段训练参数
        #   此时模型的主干被冻结了，特征提取网络不发生改变
        #   占用的显存较小，仅对网络进行微调
        # ----------------------------------------------------#
        Batch_size = 64
        Lr = 1e-3
        Init_epoch = 0
        Freeze_epoch = 50

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        model.compile(
            loss={'Embedding': triplet_loss(batch_size=Batch_size), 'Softmax': 'categorical_crossentropy'},
            optimizer=Adam(lr=Lr), metrics={'Softmax': 'categorical_accuracy'}
        )
        print('Train with batch size {}.'.format(Batch_size))

        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes, Batch_size)
        val_dataset = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes, Batch_size)

        model.fit_generator(
            generator=train_dataset,
            steps_per_epoch=epoch_step,
            validation_data=val_dataset,
            validation_steps=epoch_step_val,
            epochs=Freeze_epoch,
            initial_epoch=Init_epoch,
            use_multiprocessing=True if num_workers > 1 else False,
            workers=num_workers,
            callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history,
                       lfw_callback] if lfw_eval_flag else [checkpoint_period, reduce_lr, early_stopping, tensorboard,
                                                            loss_history]
        )

    if Freeze_Train:
        for i in range(freeze_layer):
            model.layers[i].trainable = True

    if True:
        # ----------------------------------------------------#
        #   解冻阶段训练参数
        #   此时模型的主干不被冻结了，特征提取网络会发生改变
        #   占用的显存较大，网络所有的参数都会发生改变
        # ----------------------------------------------------#
        Batch_size = 32
        Lr = 1e-4
        Freeze_epoch = 50
        Epoch = 100

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        model.compile(
            loss={'Embedding': triplet_loss(batch_size=Batch_size), 'Softmax': 'categorical_crossentropy'},
            optimizer=Adam(lr=Lr), metrics={'Softmax': 'categorical_accuracy'}
        )
        print('Train with batch size {}.'.format(Batch_size))

        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes, Batch_size)
        val_dataset = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes, Batch_size)

        model.fit_generator(
            generator=train_dataset,
            steps_per_epoch=epoch_step,
            validation_data=val_dataset,
            validation_steps=epoch_step_val,
            epochs=Epoch,
            initial_epoch=Freeze_epoch,
            use_multiprocessing=True if num_workers > 1 else False,
            workers=num_workers,
            callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history,
                       lfw_callback] if lfw_eval_flag else [checkpoint_period, reduce_lr, early_stopping, tensorboard,
                                                            loss_history]
        )
