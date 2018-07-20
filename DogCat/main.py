#coding:utf8
from config import opt
import os
import torch as t
import models
from data.dataset import DogCat
# from data import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer


def test(**kwargs):

    '''
    猫狗大战 测试集 预测完成后写入cvs中每张图片为狗的概率
    '''
    # configure model
    # 直接使用指定的module
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # data
    # 加载测试集数据
    train_data = DogCat(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []
    for ii,(data,path) in enumerate(test_dataloader):
        input = t.autograd.Variable(data,volatile = True)
        if opt.use_gpu: input = input.cuda()
        # score代表分类后的每一项的得分
        score = model(input)
        # [:,0]全行，第一列，第一列为狗，计算是狗的得分，通过softmax计算概率
        probability = t.nn.functional.softmax(score)[:,0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同（两个对象 path,probability）
        # 两个对象 path,probability  逐元素拿出来打包成一个元组，返回由这些元组组成的列表
        # 分模块迭代results
        batch_results = [(path_,probability_) for path_,probability_ in zip(path,probability) ]
        # 总的results
        results += batch_results
    # result_file 写入的文件地址
    write_csv(results,opt.result_file)
    return results

def write_csv(results,file_name):
    import csv
    # 调整为写入模式
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        # 写入标题
        writer.writerow(['id','label'])
        # 写入元组数据
        writer.writerows(results)
    
def train(**kwargs):
    '''
    训练
    '''
    # 根据命令行参数更新配置

    vis = Visualizer(opt.env)

    # step1: configure model（定义网络）
    # 将config里的model赋值给models，从而定义model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # step2: data（定义数据）
    # 加载训练集
    train_data = DogCat(opt.train_data_root,train=True)
    # 加载验证集
    val_data = DogCat(opt.train_data_root,train=False)

    # 使用dataloader加载数据
    # 加载训练集数据
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        # 打乱数据
                        shuffle=True,num_workers=opt.num_workers)
    # 加载验证集数据
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    # step3: criterion and optimizer（定义损失函数和优化器）
    criterion = t.nn.CrossEntropyLoss() # 分类问题使用交叉熵，优化器使用Adam
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)
        
    # step4: meters（计算重要指标，平滑处理之后的损失，还有混淆矩阵）
    # 计算所有meter的的平均值和标准差，统计一个ecoph中损失的平均值
    loss_meter = meter.AverageValueMeter() # 损失值
    # 混淆矩阵 2表示2分类问题
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # train（开始训练）
    for epoch in range(opt.max_epoch):

        # 清空仪表信息和混淆矩阵信息
        loss_meter.reset()
        confusion_matrix.reset()

        # 迭代训练集的加载器dataloader
        for ii,(data,label) in enumerate(train_dataloader):

            # train model （训练网络参数）
            # 输入为data
            input = Variable(data)
            # 输出target为label
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            # 优化器梯度清零
            optimizer.zero_grad()
            # 计算出输入的概率
            score = model(input)
            # 损失函数
            loss = criterion(score,target)
            # 反向传播，自动求梯度
            loss.backward()
            # 更新优化器的可学习参数
            optimizer.step()
            
            
            # meters update and visualize（更新统计指标，可视化各种指标）
            loss_meter.add(loss.data[0])
            '''
            confusionmeter 用来统计问题中的分类情况，比准确率的更加详细
            '''
            confusion_matrix.add(score.data, target.data)

            # 每print_freq次可视化loss
            if ii%opt.print_freq==opt.print_freq-1:
                vis.plot('loss', loss_meter.value()[0])
                
                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()

        # 保存模型
        model.save()

        # validate and visualize（计算验证集上的指标及可视化）
        # 验证集数据val_dataloader
        val_cm,val_accuracy = val(model,val_dataloader) # 使用验证集计算准确率，val_cm混淆矩阵
        # 可视化准确率
        vis.plot('val_accuracy',val_accuracy)
        # 当前时刻的一些信息
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))
        
        # update learning （如果损失不下降，降低学习率）
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        

        previous_loss = loss_meter.value()[0]

# 传入model val_dataloader参数
def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息，用以辅助训练
    '''
    # 置于验证模式
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.type(t.LongTensor), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    # 把模型置为训练模式
    model.train()
    # 混淆矩阵的值
    cm_value = confusion_matrix.value()
    # 准确率 = 100 * 预测正确的数量与总数的比值
    # 混淆矩阵：狗预测为狗的概率+猫预测为猫的概率
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

def help():
    '''
    打印帮助的信息： python file.py help
    打印config信息
    '''
    
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':

    train()
