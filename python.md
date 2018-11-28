# Python 的使用
## Anaconda
### 包含库:
 - Data Science IDE : Jupyter , Spyder , Jupyterlab , RStuido ，VS Code
 - Analytics & Science Computing : NumPy(数组，矩阵) , SciPy(统计，优化，线性代数，信号，图像处理) ,Numba, Pandas(数据框，序列，数据处理，绘图) , DASK
 - Visualization : Bokeh , HoloViews , Datashader , matplotlib(绘图) , statsmodel(统计检验,统计建模)
 - Machine Learning : Tensorflow , H2O.ai , theano , scikit-learn(数据变换，机器学习，交叉验证)

###  下载
下载地址：[https://www.anaconda.com/download/](https://www.anaconda.com/download/)   
使用教程：[http://docs.anaconda.com/anaconda/user-guide/getting-started/](http://docs.anaconda.com/anaconda/user-guide/getting-started/)

### 使用
#### 1. anaconda 安装包  
首选  `conda install {包名}  `   
失败后用  `conda install {包名}`

## Python 常见使用
1. 字符串
```
print('It\'s a dog!') #转义输出
print("It\'s a dog!")
print('"Lee" is name') #输出双引号
print('thank\tyou')
print(r'thank\tyou')  #原样输出
str = 'Python'
print('Py' in str)  #判断是否包含"Py"
print(str[1:4])  #字符串切片，输出 "yth"
word1 = '"world!"' 
word2 = word1.strip('"') +"  " +  word1   #去除双引号并连接 
print('word1 is %s, word2 is %s'%(word1,word2), word1 in word2)  #格式控制字符串
```

2. 数值
```
import math
res = math.log(math.e ** 2)
print(res)
print(math.log(res))
```

3. 布尔
```
True and False
not False
True or False
True + False
```

4. 时间
```
# In[]
import time
now = time.strptime('2017-08-23','%Y-%m-%d')
print(now)
print(type(now))
# In[]
import datetime
someday = datetime.date(1993,2,15)
print(someday)
anotherday =  datetime.date(2018,2,15)
deltaday = anotherday - someday
print("%s  %s"%(deltaday,deltaday.days))
print(deltaday)
```

5. None 类型与 numpy中的NaN: None 表示空类型，常用于占位，NaN不是一种数据类型表示缺失值
```
print(None)
type(None)
print(type(now))
str='hello'
print(type(str))
```

6. 类型转换
```
f1 = float(3232)
print(f1)
n1=int('2334')
print(n1)
word3 = str(n1)
print(word3)
``` 

7. 运算符
```
== +  - * / % **(指数) //(取整除，地板除)
= += -= *=  /= %= **= //=
print(9/2)      # 4.5
print(9.0/2.0)  # 4.5
print(9//2)     # 4
print(9.0//2.0) # 4.0
```




------------
@  [营养大数据 数据科学组](http://git.quietalk.cn/hnbd/data)      
[@鹏飞](http://git.hnbdata.cn/lipengfei)