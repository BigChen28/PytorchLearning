#coding:utf8
import visdom
import time
import numpy as np

class Visualizer(object):
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`或者self.function
    调用原生的visdom接口
    例如：self.text('hello visdom')
         self.histogram(t.randn(1000))
         self.line(t.arange(0,10),t.arange(1,11))
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        
        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {} 
        self.log_text = ''
    def reinit(self,env='default',**kwargs):
        '''
        可理解为再初始化，传入一个新的变量
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env,**kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        一次画多个损失图形，字典形式（name,value）如('loss',0.11)表示loss的值为0.11
        '''
        # 通过遍历d的key,value进行画图
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        '''
        一次画多个图像
        '''
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y,**kwargs):
        '''
        self.plot('loss',1.00)
        '''
        # 得到下标序号
        x = self.index.get(name, 0)
        # 作图 画出y轴，x轴，图的name
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append', # 按照append画图，不重叠
                      **kwargs
                      )
        # 下标累加1
        self.index[name] = x + 1

    def img(self, name, img_,**kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        self.vis.images(img_.cpu().numpy(),
                       win=name,
                       opts=dict(title=name),
                       **kwargs
                       )


    def log(self,info,win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        打印日志
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),\
                            info=info)) 
        self.vis.text(self.log_text,win)   

    def __getattr__(self, name):
        '''
        除自定义的plot,image,log,plot_many外，slef.function等价于 self.vis.function
        '''
        return getattr(self.vis, name)

