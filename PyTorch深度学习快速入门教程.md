# python学习中的两大法宝函数

## package(PyTorch)

![[Pasted image 20251027114939.png]]
![[Pasted image 20251027115002.png]]

# PyCharm与jupyter使用及对比
==代码是以块为一个整体运行的==

## python文件
1、python文件的块是所有行的代码

2、优点：通用 传播方便 适用于大型项目
缺点：需要从头文件
## python控制台
1、以每一行为块来运行

2、优点：显示每个变量属性
缺点：不利于代码阅读及修改
## jupyter textbook
1、打开jupyter textbook
①打开anaconda prompt 
②输入conda activate py310进入py310环境
③输入如下代码，访问E盘里面的python文件
如果文件在C盘，直接输入`jupyter notebook`
E:
cd jupyter\textbook
jupyter notebook

2、以任意行为块运行的

3、优点：有利于代码阅读及修改
缺点：环境需要配置

# PyTorch加载数据初认识：pycharm中如何读取数据
## dataset 
### 用途
提供一种方式去获取数据及其label值
1、获取每一个数据及其label
2、告诉我们总共有多少个数据

### 使用
①进入pycharm
②右击总项目：打开于->资源管理器->进入总项目
③复制我们想要传入的数据，将这份数据粘贴在总项目中
④在pycharm中找到你传入的数据，然后使用copy path 或者
copy relative path复制路径后填入代码
==windows中\要替换成双斜杠 ==
![[Pasted image 20251027150626.png]]

#### 想要获取所有图片的地址
![[Pasted image 20251027151206.png]]
![[Pasted image 20251027151350.png]]
输入文件夹的路径->`import os`->创建数组
![[Pasted image 20251027151415.png]]
利用下标访问--成功输出第一个图片的名称

![[Pasted image 20251027152207.png]]
```
from torch.utils.data import Dataset
```
- 从 PyTorch 的工具模块中导入 `Dataset` 类，这是所有自定义数据集的基类
```
class MyData(Dataset):
```
- 定义一个名为 `MyData` 的类，继承自 `Dataset`，表示你要创建一个自定义的数据集类
```
def __init__(self, root_dir, label_dir):
```
- 初始化方法，传入两个参数：
    - `root_dir`: 图像数据的根目录
    - `label_dir`: 标签目录（在你的代码中其实是图像子目录名）
```
self.root_dir = root_dir
self.label_dir = label_dir
```
- 将传入的路径参数保存为类的属性，方便后续使用
- ==即 将变量声明为类的全局变量，方便后续使用==
```
self.path = os.path.join(self.root_dir, self.label_dir)
```
- 使用 `os.path.join` 拼接出完整的图像文件夹路径，例如：`"data/train"`。
```
self.img_path = os.listdir(self.path)
```
- 获取该路径下所有文件名（通常是图像文件），==并保存为列表 `img_path`==
```
def __getitem__(self, idx):
```
- 定义获取数据集中第 `idx` 个样本的方法，这是 PyTorch `DataLoader` 会自动调用的
```
img_name = self.img_path[idx]
```
- 根据索引==获取图像文件名==，例如 `"dog1.jpg"`
```
img = Image.open(img_item_path)
```
- 使用 ==PIL 库==打开图像文件，返回一个 `Image` 对象
```
    label = self.label_dir
```
- 将 `label_dir` 作为标签返回
- 但==注意==
- 这里的标签是字符串（如 `"train"`），如果你有多个类别，建议用数字或从文件中读取更精确的标签
```
    return img, label
```
- 返回==图像==和==标签的元组==，供模型训练或验证使用
```
def __len__(self):
    return len(self.img_path)
```
- 返回数据集中==样本的总数==，也就是图像文件的数量

![[Pasted image 20251027152657.png]]
```
root_dir = "dataset/train"
```
- 定义数据集的==根目录==，通常是图像数据的主文件夹
- 例如：`dataset/train/ants/xxx.jpg` 和 `dataset/train/bees/yyy.jpg`
```
ants_label_dir = "ants"
bees_label_dir = "bees"
```
- 分别定义蚂蚁和蜜蜂图像所在的==子目录名==
[^1]- 这两个子目录是==分类标签==的代表

```
ants_dataset = MyData(root_dir, ants_label_dir)
```
- 创建一个 `MyData` 类的实例，用于加载蚂蚁图像数据
- 实际路径是：`dataset/train/ants`
```
bees_dataset = MyData(root_dir, bees_label_dir)
```
- 创建另一个 `MyData` 实例，用于加载蜜蜂图像数据
- 实际路径是：`dataset/train/bees`
```
train_dataset = ants_dataset + bees_dataset
```
- 将蚂蚁和蜜蜂两个数据集拼接成一个总的训练数据集
- 这个操作依赖于 PyTorch 的 `Dataset` 支持加法运算（即 `__add__` 方法），它会==自动合并两个数据集的样本==（这样就可以通过合并后的数据集来访问数据）

[^1]: 你现在的数据结构是这样的：
	```
	dataset/train/
	├── ants/
	│   ├── ant1.jpg
	│   ├── ant2.jpg
	├── bees/
	│   ├── bee1.jpg
	│   ├── bee2.jpg
	```
	
	这里的 `"ants"` 和 `"bees"` 文件夹名，其实就是**类别名**。你把图像放在不同的子目录里，就是在告诉模型：
	
	- `"ants"` 文件夹里的图像 → 属于蚂蚁类
	    
	- `"bees"` 文件夹里的图像 → 属于蜜蜂类
	    
	
	所以我们说：**子目录名就是分类标签的代表**

## Tensorboard
___语法：（注意传入类型）___
![[Pasted image 20251028105600.png]]
### 通过Tensorboard绘制图像
![[Pasted image 20251027191811.png]]

在终端中执行下列代码：
 tensorboard --logdir=logs --port=6007
![[Pasted image 20251027185807.png]]
如果存在 “混乱” 的情况--将logs下面的==子文件全部删除==
然后在终端中==重新执行==打开tensorboard的操作代码

### add_image()的使用
![[Pasted image 20251027194807.png]]
```
from torch.utils.tensorboard import SummaryWriter
```
- 从 PyTorch 中导入 `SummaryWriter`，这是用于写入 TensorBoard 日志的工具。
- 可以记录标量、图像、模型结构等，方便可视化训练过程

```
import numpy as np
```
- 导入 ==NumPy 库==，==常用于处理数组、矩阵等==数值数据。
- 在==图像处理==或==张量转换==中经常用到

```
from PIL import Image
```
- 从 Python ==图像库 PIL ==中导入 （import）==`Image` 类==，用于==打开==和==处理==图像文件。
    
- 支持多种格式（如 JPG、PNG），可以==转换为张量==或 ==NumPy 数组==
```
writer = SummaryWriter("logs")
```
- 创建一个 `SummaryWriter` 实例，指定[^2]==日志目录==为 `"logs"`
- 所有写入的数据（如图像、标量）都会保存在这个文件夹中，供 TensorBoard 读取
```
image_path = "data/train/ants_image/0013035.jpg"
```
- ==定义==图像文件的路径
    
- 这里是一个蚂蚁图像，位于 `data/train/ants_image/` 文件夹下
```
img_PIL = Image.open(image_path)
```
- 使用 ==PIL== 打开图像文件，返回一个 `Image` 对象。
    
- 此时图像还==不是张量，不能直接写入 TensorBoard==，需要进一步转换

[^2]: **日志目录（log directory）**就是一个文件夹，用来**保存程序运行过程中记录的数据**，比如：
	
	- 模型训练过程中的损失值、准确率
	    
	- 图像、音频、文本等可视化内容
	    
	- 网络结构、学习率变化等信息
	    
	
	这些数据被写入日志文件后，可以用工具（比如 ==TensorBoard==）来可视化分析

![[Pasted image 20251027194754.png]]
```
img_array = np.array(img_PIL)
```
- 将 `img_PIL`（一个 PIL 图像对象）==转换为 NumPy 数组==
    
- ==这是 TensorBoard 接受的图像格式==之一，通常是形如 `(H, W, C)` 的==三维数组==

```
print(type(img_array))
```
- 打印 `img_array` 的类型，确认它是 `<class 'numpy.ndarray'>`

```
print(img_array.shape)
```
- 打印图像数组的形状，比如 `(224, 224, 3)` 表示 224×224 的 RGB 图像。
- ==确认==图像是否符合 TensorBoard 的要求（==必须是三维，且通道数为 1 或 3==）

```
writer.add_image("test", img_array, 1)
```

- 将图像写入 TensorBoard，标签为 `"test"`，[^3]步数为 `1`。
    
- 你可以在 TensorBoard 的图像面板中看到这张图
```
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)
```
- 循环 100 次，记录标量值
- 标签是 `"y=2x"`，但记录的是 `y = 3x`（即 `3*i`）
- 第三个参数 `i` 是步数（step），用于==在 TensorBoard 中横轴定位==
📌 你可以改成 `2*i` 来和注释一致，或者改注释为 `# y = 3x`

```
writer.close()
```
- 关闭 `SummaryWriter`，保存所有日志数据。
- 这是一个好习惯，确保数据完整写入磁盘

[^3]: 在训练模型或记录数据时，**步数（step）**表示当前记录发生在第几次迭代、训练轮次或时间点。它是 TensorBoard 图表的横轴，用来表示数据随时间的变化
	
	这里的 `1` 就是步数，意思是：
	- 你在第 **1 步** 记录了这张图像
	- TensorBoard 会把这张图像显示在步数为 1 的位置
	**关于步数**
	![[Pasted image 20251027195540.png]]
	![[Pasted image 20251027195552.png]]
	可以直接看到训练当中给model提供了哪些数据
	或者对model进行测试的时候可以观察输出结果

## Transforms
___ctrl+鼠标点击函数（即可进入内置函数解释--比百度的解释更加权威正确）___
### Transforms的结构及用法
![[Pasted image 20251027200959.png]]

### python当中的用法->Tensor数据类型
==totensor：将 PTL image 或者 numpy.ndarraay转换成一个Tensor的数据类型==
#### transforms该如何使用呢？
![[Pasted image 20251027201552.png]]
![[Pasted image 20251027201604.png]]
注意：transforms.ToTensor(==传入[^4]pic类型变量==)
所以上述代码中将img这个pic类型变量传入ToTensor
![[Pasted image 20251027201951.png]]
**Totensor需要我们传入一个pic，被input赋值给result，作为输入量，通过tool这个工具输出结果**

[^4]: picture类型
#### 为什么需要Tensor这个数据类型
![[Pasted image 20251027202656.png]]
`writer.add_image()` 是 PyTorch 中用于将图像写入 TensorBoard 的方法，属于 `SummaryWriter` 类的一部分，里面的参数需要一个“名字”加上一个torch_Tensor的数据类型（可以通过长按ctrl+点击类名来知道所需类型）[^5]![[Pasted image 20251027202948.png]]

[^5]: 
	Tensor_img就是图片的名称




### 常见的transforms
![[Pasted image 20251027203133.png]]
#### python中内置函数__call__的用法
下划线__表示内置函数
![[Pasted image 20251028095624.png]]
内置函数就表示不需要用 . 的方式访问类中的函数，直接用（）+参数的形式即可调用
#### ToTensor的使用
![[Pasted image 20251028100314.png]]
___将PIL的img通过transforms.ToTensor转变为一个tensor数据类型的img，那么此时这个img就可以放进Tensorboard里面___

#### ToPILimage的使用
![[Pasted image 20251028100551.png]]
将tensor或者ndarray数据类型转换为PIL

#### Normalize的使用
输入参数分别为：均值（mean）+标准差（std）
![[Pasted image 20251028101234.png]]
**计算过程**：input[channel]=(input[channel]-mean[channel])/std[channel]
![[Pasted image 20251028102317.png]]

#### Resize使用（改变尺寸--缩放）
##### resize
![[Pasted image 20251028102036.png]]
___需要一个PIL类型的输入量___：上述totensor定义的img就是一个PIL类型，在此处直接引用了
![[Pasted image 20251028102503.png]]
输出结果为：（原来的尺寸是3200 * 1800，现在变成512 * 512）
![[Pasted image 20251028102122.png]]

##### compose--resize
compose要求：按顺序传入两个参数
参数1的**输出**类型，必须与参数2的**输入**类型匹配
![[Pasted image 20251028103032.png]]

#### RandomCrop
**真** 随机裁剪
![[Pasted image 20251028103411.png]]

### 总结
![[Pasted image 20251028103759.png]]
![[Pasted image 20251028103820.png]]

## torchvision中的数据集（dataset）使用
### 一些标准数据集
### CIFAR-10
___The CIFAR-10 dataset consists of 60000 32 * 32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.___
![[Pasted image 20251028104736.png]]

参数 `download=True` 的意思是：**如果指定路径（这里是** `"./dataset"`**）下没有找到 CIFAR-10 数据集，就自动从网上下载这个数据集并保存到该路径**
![[Pasted image 20251028104842.png]]
**打印** `test_set` **数据集中第一个样本的内容**
![[Pasted image 20251028105703.png]]
`torchvision.transforms.Compose([...])`：把多个图像处理步骤组合在一起，形成一个“流水线”
`ToTensor()`：把原始图像（通常是 PIL 图像或 NumPy 数组）转换成 PyTorch 的张量（Tensor），并且会自动把像素值从 `[0, 255]` 缩放到 `[0.0, 1.0]` 的范围
然后修改train_set 和 test_set
![[Pasted image 20251028105913.png]]
**加载 CIFAR-10 数据集，并对每张图像应用预处理操作** `ToTensor()`**，把它转换成 PyTorch 张量格式**
`transform=dataset_transform` 是关键，它表示每张图像在被加载时都会自动执行之前定义的转换`dataset_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])`

#### captions
![[Pasted image 20251028110147.png]]

### dataloader
- **Dataset 就像是厨房准备好的餐盒**，里面装着一份份数据（比如图像和标签），你可以按编号取出每一份。
    
- **DataLoader 就像是一个送餐机器人**，它负责：
    
    - 按批次（batch）把数据打包送给你
        
    - 自动打乱顺序（shuffle）让训练更有效
        
    - 多线程加速送餐（num_workers）
        
    - 每次送你一批，不用你自己去厨房拿


![[Pasted image 20251028113235.png]]
`test_data = torchvision.datasets.CIFAR10(
    `"./dataset",`                      # 数据保存路径
    `train=False,`                      # 加载测试集（不是训练集）
	`transform=ToTensor())`把每张图像从 PIL 格式转换成 PyTorch 的张量，并把像素值从 `[0, 255]` 缩放到 `[0.0, 1.0]`
**用 `DataLoader` 打包数据**	
`test_loader = DataLoader(
    `dataset=test_data,`     # 数据来源
    `batch_size=4,`          # 每次从dataset中取 4 张图像
    ![[Pasted image 20251028113626.png]]
    ___dataloader中的返回：
    ①将img系列打包成imgs，返回
    ②将target系列打包成targets，返回___
    `shuffle=True,`          # 打乱顺序（测试时一般设为 False）
    `num_workers=0,`         # 不使用额外线程加载数据（适合初学者）
    `drop_last=False)`        # 保留最后一批（即使不满 4 个）
![[Pasted image 20251028114501.png]]
- `test_data[0]`：取出测试集中的第一张图像和它的标签
    
- `img.shape`：查看图像的张量形状（3 个通道，32×32 像素）
    
- `target`：图像的类别编号（如 0 表示飞机）

- ==创建一个 TensorBoard 写入器==，日志会保存在 `"dataloader"` 文件夹中
    
- 后面你可以用命令 `tensorboard --logdir=dataloader` 打开可视化界面
[^6]
- 用 `test_loader` 一批一批地加载图像数据

- `writer.add_images(...)` 会把这一批图像写入 TensorBoard
    
- `step` 是记录的步数，每一批图像对应一个时间步

[^6]: - 将 `test_loader` 想象成一个“送快递的机器人”，它把数据集打包成一批批送过来
	- 每次 `data` 是一个元组 `(imgs, targets)`，包含：
	- `imgs`：一批图像（比如 4 张）
	- `targets`：对应的标签（比如 `[0, 3, 5, 1]`）

# 神经网络的搭建
## nn.Module的使用
![[Pasted image 20251028115401.png]]
![[Pasted image 20251028120647.png]]
```
class Tudui(nn.Module):
```
- 定义一个名为 `Tudui` 的类，继承自 PyTorch 的 `nn.Module`，这是所有神经网络模块的基类
```
def __init__(self):
        super().__init__()
```
- 构造函数 `__init__`：初始化这个模块。
    
- `super().__init__()`：调用父类 `nn.Module` 的初始化方法，**必须写**，否则模块功能不完整
```
def forward(self, input):
        output = input + 1
        return output
```
- 定义模块的前向传播逻辑（即模型的计算过程）
    
- `input + 1`：把输入张量加 1，作为输出
    
- `forward()` 是 PyTorch 中的标准方法，**模型调用时会自动执行它**
```
tudui = Tudui()
```
- 创建一个 `Tudui` 模型实例，准备使用
```
x = torch.tensor(1.0)
```
- 创建一个==张量 `x`==，值为 `1.0`，这是你要送进模型的输入
```
output = tudui(x)
```
- ==把张量 `x` 送进模型 `tudui`==，会自动调用 `forward()` 方法。
    
- 实际执行的是：`output = x + 1`，所以结果是 `2.0`

# 卷积操作
**CONV2D**
![[Pasted image 20251028151111.png]]
前面几个参数需要我们自己设置，后面的参数都已经默认设置好了
![[Pasted image 20251028142928.png]]
![[Pasted image 20251028143142.png]]
调用**卷积函数** 计算卷积后的输出
①stride
![[Pasted image 20251028143423.png]]
运行结果
![[Pasted image 20251028143555.png]]
stride = 1与stride =2 计算方式不一样
![[Pasted image 20251028143520.png]]
运行结果
![[Pasted image 20251028143536.png]]
②pudding
padding=1时，在输入图像上下左右两边添加“一个像素”
![[Pasted image 20251028144141.png]]
**空的地方默认为0，卷积核按照原先方式移动并计算**
具体计算过程如下：
![[PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】 2025-10-28 15-06-46 1.mp4]]

# 神经网络
## 卷积层
### 卷积层计算函数
![[Pasted image 20251028151111.png]]
前面几个参数需要我们自己设置，后面的参数都已经默认设置好了
___在卷积神经网络中，`kernel_size` 表示卷积核的尺寸，也叫滤波器大小。它决定了每次卷积操作覆盖图像的区域大小___
![[Pasted image 20251028151618.png]]
#### stride 和 pudding
no stride表示stride=1
![[PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】 2025-10-28 15-12-52.mp4]]
#### in_channel 和  out_channel
![[Pasted image 20251028151714.png]]
In_channel=1和Out_channel=1的时候，就只有一个卷积核 在输入图像里面卷积
Out_channel=2的时候，卷积层会生成两个**不一定**一样大的卷积核来对输入图像进行卷积，从而生成两个**卷积后的输出**
![[Pasted image 20251028154014.png]]
```
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
```

- 定义一个名为 `Tudui` 的神经网络类，继承自 `nn.Module`。
    
- `__init__` 是初始化方法，调用父类构造函数。
    
- `self.conv1` 是一个卷积层：
    
    - 输入通道数为 3（RGB 图像）**输入图像**
        
    - 输出通道数为 6（提取 6 个特征图）**卷积后的输出**
        
    - 卷积核大小为 3×3
        
    - 步长为 1，表示每次滑动一个像素 stride =1
        
    - ==无填充，意味着输出尺寸会比输入小==  pudding=0
```
def forward(self, x):
        x = self.conv1(x)
        return x
```

- 定义前向传播逻辑。
    
- 输入 `x` 是==图像张量==，经过卷积层处理后返回结果
```
output = tudui(images)
```

- 将图像输入模型 `tudui`，得到卷积后的输出赋给output
    
```
print(output.shape)
    # torch.Size([4, 6, 30, 30])
```

- 打印输出张量的形状：
    
    - 表示 batch 中有 4 张图像（前面dataloader里面定义好的）
        
    - 每张图像被卷积后变成 6 个通道
        
    [^7]- 尺寸为 30×30（说明输入图像尺寸可能是 32×32）
```
writer.add_images("input", imgs, step)
```

- 将原始图像写入 TensorBoard，标签为 `"input"`。
    
- TensorBoard 会显示这些图像，便于观察输入数据
```
output = torch.reshape(output, (-1, 3, 30, 30))
```

- 将输出张量 reshape 成 `(batch_size * 通道数 / 3, 3, 30, 30)` 的形式。
    
- [^8]这是为了将输出转换为 3 通道图像，便于 TensorBoard 显示
    
- ⚠️ 但原始输出是 6 通道，直接 reshape 成 3 通道可能会导致图像混乱或信息丢失
```
 writer.add_images("output", output, step)
```

- 将 reshape 后的输出图像写入 TensorBoard，标签为 `"output"`。
    
- 如果 reshape 不合理，TensorBoard 显示的图像可能不具备可读性
```
step = + 1
```

- ⚠️ 语法错误：`step = +1` 实际上是将 `step` 赋值为正数 1，而不是递增


![[Pasted image 20251028152824.png]]

## 最大池化的使用 MaxPool2d
用处：表达数据的基本意思的同时，又能减少存储空间，能减少数据量，加快训练

![[Pasted image 20251028155208.png]]
#### 池化层的默认值=kernel_size
#### dilation
![[QQ20251028-1616.mp4]]
#### ceil_mode
![[Pasted image 20251028155604.png]]
![[QQ20251028-155841.mp4]]
### 代码实现
![[屏幕截图 2025-10-28 190413.png]]
![[Pasted image 20251028190432.png]]
```
input = torch.tensor([[ 
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 4],
    [3, 4, 5, 6, 7]
]])
```

- 创建一个[^9]形状为 `(1, 5)` 的二维列表，表示一个 3×5 的图像或特征图。
    
- 外层的 `[[...]]` 是为了构造一个 3 行 5 列的==二维==张量
```
input = torch.reshape(input, (-1, 1, 3, 5))
print(input.shape)
```

- `reshape` 将张量变为形状 `(1, 1, 3, 5)`：
    
    - `-1` 表示==自动推断 batch size==（这里是 1）
        
    - `1` 表示通道数（channel），适用于灰度图
        
    - `3` 和 `5` 是高和宽
        
- 打印结果是：`torch.Size([1, 1, 3, 5])`，符合 `MaxPool2d` 的输入格式 `(N, C, H, W)`

```
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)
```

- 创建一个名为 `Tudui` 的神经网络类，继承自 `nn.Module`
    
- `__init__` 是构造函数，初始化最大池化层：
    
    - `kernel_size=3`：池化窗口大小为 3×3
        
    - `ceil_mode=False`：使用向下取整计算输出尺寸（默认行为）

```
 def forward(self, input):
        output = self.maxpool1(input)
        return output
```

- `forward` 方法==定义了输入如何通过网络处理==
    
- 将输入张量传入 `maxpool1` 层，返回池化后的结果

```
tudui = Tudui()
output = tudui(input)
print(output)
```

- 创建 `Tudui` 类的实例
    
- 将之前构造的 `input` 张量传入网络，得到池化结果
    
- 打印输出张量，形状为 `(1, 1, 1, 3)`，因为：
    
    - 输入尺寸是 `(3, 5)`
        
    - 使用 `3×3` 池化窗口，步长默认为 3（如果没指定）

## 非线性激活

### RELU
![[Pasted image 20251028191355.png]]
![[Pasted image 20251028191443.png]]
***代码实现***
![[Pasted image 20251028191849.png]]
***为什么要执行 reshape***
在 PyTorch 中，大多数神经网络层（比如卷积、池化、激活）都要求输入张量的形状是 **四维** 的：
（N, C, H, W)

- **N**：batch size（样本数量）（dataloader）
    
- **C**：channel 数（通道数，比如灰度图是 1，RGB 是 3）
    
- **H**：高度（图像或特征图的==行数==）
    
- **W**：宽度（图像或特征图的==列数==）
**原始张量形状**
```
input = torch.tensor([[1, -0.5],
                      [1, 3]])
```
这个张量的形状是 `(2, 2)`，也就是一个二维矩阵，没有 batch 和 channel 的维度

**reshape的作用**
```
input = torch.reshape(input, (1, 1, 2, 2))
```
这一步将张量变成形状 `(1, 1, 2, 2)`，含义如下：
- `1`：只有一个样本（batch size）
- `1`：只有一个通道（channel）
- `2 × 2`：图像的高和宽
✅ 这样就符合了 PyTorch 层的输入要求，才能被 `ReLU` 或其他层正常处理

### sigmoid
![[Pasted image 20251028191507.png]]
![[Pasted image 20251028191520.png]]


## 线性层及其他层的介绍
### BatchNorm2D
![[Pasted image 20251028193338.png]]
![[Pasted image 20251028193349.png]]

### Linear Layers
传入参数（参数1：input    参数2：output）
![[Pasted image 20251028201405.png]]
![[Pasted image 20251028201448.png]]
![[Pasted image 20251028201458.png]]
```
    self.linear1 = Linear(196608, 10)
```
- 定义一个线性层（全连接层），输入维度是 196608，输出维度是 10。
- 这意味着模型期望输入是一个形状为 `[batch_size, 196608]` 的张量，输出是 `[batch_size, 10]`
```
def forward(self, input):
        output = self.linear1(input)
        return output
```

- 定义前向传播逻辑：输入数据通过 `linear1` 层，得到输出。
    
- `forward` 方法是 PyTorch 模型的核心，定义了数据如何流经网络



## 搭建小实战和Sequential的使用
### CIFAR-10模型结构
![[Pasted image 20251028203714.png]]
___pay attention：
CIFAR-10标准数据集是图像分类任务的输入数据，而CIFAR-10模型结构是为处理这些数据而设计的神经网络架构，两者分别代表“数据”和“处理方法”___
#### 🔍 区别（CIFAR-10标准数据集/CIFAR-10模型结构）

|项目|CIFAR-10数据集|CIFAR-10模型结构|
|---|---|---|
|本质|图像数据和标签|神经网络架构|
|作用|提供训练和测试数据|学习数据特征并分类|
|内容|60,000张图像 + 标签|卷积层、池化层、全连接层等|
|变化性|固定不变|可自由设计和优化|
|使用方式|`torchvision.datasets.CIFAR10` 加载|`torch.nn.Module` 定义模型类|

|**简单CNN**|2~3个卷积层 + 池化层 + 全连接层|
![[Pasted image 20251028204540.png]]
代码实现：
![[Pasted image 20251028205103.png]]
![[Pasted image 20251028205115.png]]
这个模型叫做 `Tudui`，它的结构如下：

1. 三个卷积层（`Conv2d`）+ 三个最大池化层（`MaxPool2d`）
    
2. 一个展平层（`Flatten`）
    
3. 两个全连接层（`Linear`）

```
[^11]self.linear1 = Linear(1024, 64)
self.linear2 = Linear(64, 10)
```
- 两个全连接层：
    - 第一个将 1024 个特征压缩为 64 个
    - 第二个将 64 个特征映射到 10 个类别（CIFAR-10）

### Squential
利用squential可以简化代码
![[Pasted image 20251028210635.png]]
可以简化成：
![[Pasted image 20251028210700.png]]
利用tensorboard可视化
![[Pasted image 20251028210838.png]]
![[Pasted image 20251028210849.png]]


[^7]: 卷积操作本质上是用一个小窗口（卷积核）在图像上滑动并计算局部特征。每次滑动时，它只覆盖图像的一部分区域，所以输出尺寸会变小，除非使用了填充（padding），填充输入图像从而使输出图像像素不变

[^8]: TensorBoard 的 `add_images()` 方法用于可视化图像张量，但它对图像的通道数有要求：
	
	|通道数|显示方式|
	|---|---|
	|1|显示为灰度图像|
	|3|显示为彩色图像（RGB）|
	|>3|只显示前 3 个通道，或报错|
	
	所以，如果你传入的是 6 通道的张量（比如卷积输出），TensorBoard：
	
	- 可能只显示前 3 个通道
	    
	- 或者直接报错，提示通道数不支持
[^9]: ```
	[
	  [  # 第一维：长度为1
	    [1, 2, 3, 0, 1],  # 第二维：长度为3，每个元素是一个长度为5的列表
	    [0, 1, 2, 3, 4],
	    [3, 4, 5, 6, 7]
	  ]
	]
	```
	这就构成了一个 **二维张量**，形状是 `(1, 3, 5)`，也就是：
	- 第一维：1 个样本（batch）
	- 第二维：3 行
	- 第三维：5 列
	但如果你用 `torch.tensor([[1, 2, 3, 4, 5]])`，那就是形状 `(1, 5)`，因为只有一层嵌套
	
	运行下面的代码验证
	```
	x = torch.tensor([[ 
	    [1, 2, 3, 0, 1],
	    [0, 1, 2, 3, 4],
	    [3, 4, 5, 6, 7]
	]])
	print(x.shape)  # 输出：torch.Size([1, 3, 5])
	```
	
	- `torch.tensor([[1, 2, 3, 4, 5]])` → `(1, 5)`
	- `torch.tensor([[[1, 2, 3], [4, 5, 6]]])` → `(1, 2, 3)`
	- 每多一层嵌套，就多一维

[^10]: 你完全可以只用一个全连接层，把展平后的 1024 个特征直接映射到 10 个类别：
	
	python
	
	```
	self.linear = nn.Linear(1024, 10)
	```
	
	这样在 `forward` 中就可以直接：
	
	python
	
	```
	x = self.flatten(x)
	x = self.linear(x)
	```

[^11]: 你完全可以只用一个全连接层，把展平后的 1024 个特征直接映射到 10 个类别：
	```
	self.linear = nn.Linear(1024, 10)
	```
	
	这样在 `forward` 中就可以直接：
	```
	x = self.flatten(x)
	x = self.linear(x)
	```
	
	加一个中间层的原因：为了提高模型的**表达能力**和**学习能力**
	
	|结构|优点|缺点|
	|---|---|---|
	|**一个全连接层：1024 → 10**|简单、参数少、训练快|表达能力弱，可能欠拟合|
	|**两个全连接层：1024 → 64 → 10**|增加非线性、提升拟合能力|参数稍多，训练稍慢|
	
	中间的 `64` 是一个**隐藏层**，它可以帮助模型学习更复杂的特征组合。尤其是在图像分类任务中，图像特征往往不是线性可分的，所以多一层可以让模型更“聪明”


# 损失函数与反向传播
## Loss Functions
**越小越好**
![[Pasted image 20251029190901.png]]
![[Pasted image 20251029191017.png]]
N：[^12]**Batch size（批大小）** 指的是在一次前向传播和反向传播中，模型处理的训练样本数量
![[Pasted image 20251029191324.png]]
L1.loss（）括号里面可以填计算方式 下图表示“相加”
![[Pasted image 20251029191403.png]]

## CrossEntropyLoss
![[Pasted image 20251029192413.png]]
![[Pasted image 20251029192430.png]]

## 反向传播
![[Pasted image 20251029193416.png]]
![[QQ20251029-193448.mp4]]
![[QQ20251029-193559.mp4]]
### 优化器
![[Pasted image 20251029194520.png]]
```
for input, target in dataset:
```
🔁 **遍历数据集**
- `dataset` 是一个可迭代对象，通常是 `DataLoader`。
- 每次迭代取出一批数据：`input` 是输入特征，`target` 是对应标签

```
optimizer.zero_grad()
```

🧼 **清除旧的梯度**
- 每次反向传播前都要清零梯度，否则会累加上一次的结果。
- 就像每次考试前都要擦掉黑板上的旧答案
- optimizer是我们所定义的优化器
- zero_grad()是清零函数

```
 output = model(input)
```
📡 **前向传播**
- 把输入数据喂给模型，得到预测结果 `output`。
- 模型内部会执行一系列线性变换和激活函数

```
 loss = loss_fn(output, target)
```
📉 **计算损失**
- `loss_fn` 是损失函数，比如 `nn.CrossEntropyLoss()` 或 `nn.MSELoss()`。
- 它衡量预测值 `output` 和真实值 `target` 的差距

```
 loss.backward()
```
🔁 **反向传播**
- 自动计算每个参数对损失的梯度。
- PyTorch 会根据计算图自动完成链式求导

```
optimizer.step()
```
🚶 **更新参数**
- 根据刚才算出的梯度，优化器（如 Adam、SGD）更新模型的权重。
- 这一步是“学习”的关键，让模型变得更聪明

优化器的定义
![[Pasted image 20251029195451.png]]
==lr是学习速率==

### 优化器＋模型训练
![[Pasted image 20251029200210.png]]
```
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
```
🔧 **定义优化器**
- 使用随机梯度下降（SGD）优化器
- `tudui.parameters()` 表示模型的所有可训练参数
- `lr=0.01` 是学习率，控制每次参数更新的步长

```
for epoch in range(2):
```
🔁 **训练轮数循环**
- 训练 2 个 epoch（遍历数据集两次）
- 每次 epoch 都会完整训练一遍数据集

```
outputs = tudui(imgs)
```
📡 **前向传播**
- 将图像输入模型，得到预测结果 `outputs`。
    
```
        result_loss = loss(outputs, targets)
```
📉 **计算损失**
- 用交叉熵损失函数计算预测值与真实标签的误差

```
optim.step()
```
🚶 **更新参数**
- 根据梯度，优化器更新模型的[^13]权重，让模型更“聪明”

```
running_loss = running_loss + result_loss
```
📊 **累计损失**
- 把当前 batch 的损失加到 `running_loss` 中，用于观察训练效果

```
print(running_loss)
```
📢 **输出总损失**
- 打印当前 epoch 的总损失，帮助你判断模型是否在收敛


[^12]: 假设你有一个包含 10,000 张图片的数据集：
	
	- 如果 **batch size = 32**，那么每次训练会从数据集中取出 32 张图片进行一次模型更新。
	    
	- 这意味着一个 epoch（遍历整个数据集一次）会有：
	    
	
	10000/32≈313 次参数更新


[^13]: 神经网络的目标是：**输入 → 预测 → 和真实值尽量接近**。 而“权重”就是模型用来做判断的“参数”，它决定了每一层如何处理数据
	
	🔁 为什么更新权重有效？
	因为每次更新都在做这几件事：
	1. **找到误差来源**：哪些权重导致预测错了？
	2. **计算每个权重的“责任”**：用梯度衡量它对误差的影响
	3. **调整方向和幅度**：朝着让误差变小的方向改动权重
	    
	这就像你在刷题时不断复盘错题，优化自己的解题策略

# 现有网络模型的使用及修改
![[Pasted image 20251029204556.png]]
```
import torchvision
```
📦 **导入 torchvision 库**
- 这是 PyTorch 的视觉工具包，包含模型、数据集和图像处理工具

```
from torch import nn
```
🧠 **导入神经网络模块**
- `nn` 是 PyTorch 中构建神经网络的核心模块

```
vgg16_false = torchvision.models.vgg16(pretrained=False)
```

📦 **加载未预训练的 VGG16 模型**

- `pretrained=False` 表示模型参数是随机初始化的

```
vgg16_true = torchvision.models.vgg16(pretrained=True)
```

📦 **加载预训练的 VGG16 模型**

- `pretrained=True` 表示使用在 ImageNet 上训练好的参数。
    
- 适合迁移学习，能加快收敛速度

```
vgg16_true.classifier.add_module('add_Linear', nn.Linear(1000, 10))
```

🔧 **在预训练模型后面加一层线性层**

- 原始 VGG16 输出是 1000 类（ImageNet），这里加一层输出 10 类（CIFAR-10）。
    
- `add_module` 会在原来的结构后追加一层

```
vgg16_false.classifier[6] = nn.Linear(4096, 10)
```

🔧 **替换未预训练模型的最后一层**

- VGG16 的 `classifier[6]` 是最后的全连接层，原本输出 1000 类。
    
- 这里直接替换为输出 10 类，适配 CIFAR-10



##### 🧠 模型与数据集的关系：谁为谁服务？

- **数据集** 是现实世界的信息集合，比如图像、文本、语音等。
    
- **模型** 是用来“理解”这些数据的数学结构，它的目标是从数据中学习规律。
    

> 简单说：**数据集是原材料，模型是加工厂。**

 🔧 为什么要“适配”？
因为不同的数据集有不同的特点，而模型的结构必须能处理这些特点：

|数据集特点|模型需要适配的地方|
|---|---|
|图像大小不同（如 CIFAR10 是 32×32，ImageNet 是 224×224）|模型输入层尺寸要匹配|
|类别数量不同（如 CIFAR10 有 10 类，ImageNet 有 1000 类）|输出层维度要改|
|数据分布不同（如灰度图 vs 彩色图）|卷积层通道数要调整|
|任务不同（分类 vs 回归）|损失函数和输出激活函数要变|

 🎯 举个例子：VGG16 模型迁移到 CIFAR10
- VGG16 原本是为 ImageNet 设计的，输出层是 1000 个神经元
- CIFAR10 只有 10 个类别，所以你必须把输出层改成 10 个神经元
- 否则模型就会“答非所问”，无法正确学习任务
    

🧩 再打个比方
你有一个万能遥控器（模型），但每个电视（数据集）红外接口不一样：
- 如果不适配红外协议，你按按钮电视不会响应
- 一旦适配好了，遥控器就能控制电视，完成任务



# 网络模型的保存与存取
![[Pasted image 20251029210312.png]]
#### 方式1：不仅保存了网络模型结构，也保存了网络模型的内部参数
方式1 保存模型
![[Pasted image 20251029210336.png]]
方式1 加载模型
![[Pasted image 20251029210600.png]]
#### 方式2：保存了网络模型的参数（以字典形式保存）
方式2 保存模型
![[Pasted image 20251029210714.png]]
方式2 加载模型
![[Pasted image 20251029210705.png]]


# 完整的模型训练套路
## 训练数据集or测试数据集
### 📦 训练数据集（Training Set）

- ✅ **作用**：用于训练模型，调整参数（权重和偏置）
    
- 🔁 **过程**：模型反复学习这些数据，通过反向传播不断优化
    
- 🧠 **比喻**：就像你刷题时反复练习的题库，目的是掌握知识
    

### 🧪 测试数据集（Test Set）

- ✅ **作用**：用于评估模型的泛化能力
    
- 🚫 **不能参与训练**：测试集上的数据模型从未见过
    
- 🧠 **比喻**：就像考试题，检验你是否真的学会了，而不是死记硬背

 ***🔍 为什么要分开？***

| 原因     | 解释                      |
| ------ | ----------------------- |
| 防止过拟合  | 如果只在训练集上表现好，可能只是“记住了答案” |
| 检验泛化能力 | 测试集能反映模型在新数据上的表现        |
| 模拟真实场景 | 现实中模型面对的是“没见过的数据”       |
***🧪 举个例子：你训练一个猫狗分类器***

- **训练集**：包含 1000 张猫和狗的图片，模型用来学习特征
    
- **测试集**：包含另外 200 张猫和狗的图片，模型从未见过
    
- 如果模型在测试集上也能准确分类，说明它真的“学会了

## 完整模型训练套路
准备数据集
![[Pasted image 20251029212741.png]]
![[Pasted image 20251029212759.png]]

搭建神经网络（单独放在另一个python文件中，注意要和主文件在同一个文件夹下面）
![[Pasted image 20251029212341.png]]
![[Pasted image 20251030085341.png]]

创建网络模型
![[Pasted image 20251029212818.png]]
损失函数
![[Pasted image 20251029212923.png]]
优化器
![[Pasted image 20251029213003.png]]
设置训练网络的一些参数
![[Pasted image 20251029213121.png]]
开始训练
![[Pasted image 20251030084858.png]]
![[Pasted image 20251030084932.png]]
![[Pasted image 20251030084943.png]]
![[Pasted image 20251030084834.png]]

# GPU训练

# 完整模型验证（测试，demo）套路
利用已经训练好的模型，然后给它输入
#### 相对路径
![[QQ20251030-91656.mp4]]
