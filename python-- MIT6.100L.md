# lecture1 个人总结
debug：调试
## 错题集锦
#### functions+循环
`def first_to_last_diff(s, c):
    `""" s is a string, c is single character string
        `Returns the difference between the index where c first
        `occurs and the index where c last occurs. If c does not
        `occur in s, returns -1.
    `"""
![[Pasted image 20251102193524.png]]

> [!NOTE] Title
> `for j in range(len(s)-1, -1, -1):`
> **`len(s)-1`**: 起始索引（最后一个元素）
> **`-1`**: 结束索引（循环在到达 -1 之前停止）
> **`-1`**: 步长（每次迭代递减 1）




# lecture 2 string io and 分支判断
## string
**注意：在对字符串进行数乘运算时，只会重复输出n次字符串，而不是进行运算**
当我想要创建一个string，我只需要将创建内容用” “括起来即可
### test01
`a = 'me'  （单引号双引号不碍事）`
`b = "myself"`  
`c = a + b`  
`print(c)  结果memyself`
### test02
s=`"abc"`
s[-1]   c
s[-2]   b
s[-3]   a
### test03 [  start:stop:step  ] 
s = `"abcdefgh"`
s[3:6]     `"def"`(从索引3开始，不包括索引6)
s[3:6:2]   `"df"`
s[:]           `"abcdefgh"`
s[::-1]       `"hgfedcba"`（逆序获取字符串）
s[4:1:-2]   `'ec'` （step<0,逆序获取字符串）
s[6:3]       `" "`(从6开始，到3结束，空字符串，不能这样输入)
s[0:-3]      `"abcde"`（从索引0开始，到倒数第3个元素之前结束，不包含倒数第3个元素）
s[-3:]         `"fgh`（表示获取最后三个元素）
### test04
一旦数据被创建，便不能修改
`s = "car"
`#s[0] = 'b'  # this is an error`
`s = 'b'+s[1:len(s)] #可以这样修改`
## io(input/output)
### test01（output）
在print语句中，在括号内 可以利用逗号/+号实现数据连接
`a = "the”
`b = 3
`c = "musketeers"
`# print(a, b, c)
`# print(a + b + c)   # this is an error(数据类型不匹配)
`print(a + str(b) + c)
`print(a, str(b), c)  #anthor way

### test02（input）
input：在等待用户的输入
![[Pasted image 20251031160705.png]]
num1：对我们来说这是一个数字，但对python来说这依旧是一个字符串，所以print结果：33333
![[Pasted image 20251031161701.png]]
Python 中，表达式 `5 * (cin, ' ')` 是在对一个元组 `(cin, ' ')` 进行乘法操作：
```
(cin, ' ')  # 是一个元组，包含两个元素
5 * (cin, ' ')  # 会重复这个元组 5 次，得到一个新元组
so print result  ('run', ' ', 'run', ' ', 'run', ' ', 'run', ' ', 'run', ' ')
```

### f-string
![[Pasted image 20251031162830.png]]
solution：将数据转换为字符串，用＋号连接，这样可以避免逗号所生成的空间导致不美观
![[Pasted image 20251031163005.png]]
 `print(f'{num*fraction} is {fraction*100}% of {num})`
### 分支判断
if 条件表达式:
    代码块
- 条件表达式的结果为 `True` 时，执行代码块。
- **缩进非常重要，通常为 4 个空格**
You Try It 1: Write a program that:
* Saves a secret number.
* Asks the user for a number guess
* Prints a bool depending on whether the guess matches the secret.
`# your code here
`# secret = 7
`# guess = int(input("Guess a number between 0 and 10: "))
`# print(secret == guess)#it was amazing!直接利用==判断，输出False/True

You Try It 2
①try making the line: elif x <= y:  
elif时承接在if下的，如果if满足，程序进入if的分支，就不会再次进入elif的分支，此时输出 M
②下列程序输出：M i
`answer =""  
`x = 11  
`y = 11  
`if x == y:  
   `answer = answer + 'M'  
`if x <= y:  
    `answer = answer + 'i'  
`else:  
    `answer = answer + 'T'  
`print(answer)

# lecture 3 迭代

# lecture 4基于字符串循环，猜测与检查，二进制01
![[Pasted image 20251031172427.png]]
## 基于字符串的循环

> [!NOTE]
> for的step必须是整数（所以approximation的时候用for循环不合适）

![[Pasted image 20251031175147.png]]
第三种写法（更加符合python语言）
`s = "demo loops - fruit loops"
 `for char in s: #遍历s中的每一个字符
     `if char in 'iu': ` `# 使用in关键字来检测当前的字符（char）是否在某个字符序列（iu）中 
         `print("There is an i or u")
 
 You Try It 1:（Smart的思路！！！）
 Assume you are given a string of lowercase letters in variable s.
 Count how many unique letters there are in s. For example, if
 s = "abca" Then your code prints 3.
`s="abhjgdbeakjjcfekhv"  
`seen=" "  
`for char in s:  
    `if char not in seen:  #not in 的使用
        `seen+=char  
`print(len(seen))

## 猜测与检查=穷举枚举（guess and check）
### python中缩进的魅力
①
`num =1  
`for j in range(10):  
    `if(num==j):  
        `print("found")  
`else:  
        `print("not found")
输出结果：
found
not found
原因：else与if没有对齐，系统会自主创造一个else与上面的if平齐，而下面的else与for循环配对（for-else组合），直接输出[^1]
②
`num =1  
`for j in range(10):  
    `if(num==j):  
        `print("found")  
    `else:  
        `print("not found")
输出结果：
not found
found
not found
not found
not found
not found
not found
not found
not found
not found
③**改进代码（算法思维！！！）**
![[QQ2025111-91529.mp4]]

### 算法思维
this code is very slow for large numbers!
`for alyssa in range(11):
     `for ben in range(11):
        `for cindy in range(11):
            `total = (alyssa + ben + cindy == 10)
            `two_less = (ben == alyssa-2)
            `twice = (cindy == 2*alyssa)
            `if total and two_less and twice:
               `print(f"Alyssa sold {alyssa} tickets")
               `print(f"Ben sold {ben} tickets")
               `print(f"Cindy sold {cindy} tickets")

  

 this code is better -- only one loop!
 `for alyssa in range(11):
    `ben = max(alyssa-2,0)
    `cindy = alyssa*2
     `if ben + cindy + alyssa == 10:
         `print(f'Alyssa sold {alyssa} tickets')
         `print(f'Ben sold {ben} tickets')
         `print(f'Cindy sold {cindy} tickets')

### 浮点误差
![[Pasted image 20251101101508.png]]
![[Pasted image 20251101101522.png]]
![[Pasted image 20251101101544.png]]
![[Pasted image 20251101101552.png]]
![[Pasted image 20251101101608.png]]
![[Pasted image 20251101101617.png]]
![[Pasted image 20251101101622.png]]

[^1]: 在Python中，`for-else` 的含义是：
	- 如果循环**正常完成**（没有被break中断），则执行else部分
	- 如果循环被break中断，则不执行else部分

## lecture5 Floats and Approximation Methods
10进制->2进制（代码）
`x=int(input("please give me a number"))  
`result=""  
`while x>0:  
    `result=str(x%2)+result  
    `x=x//2  
`print(result)
### floats
![[QQ2025111-10363.mp4]]
![[QQ2025111-103732.mp4]]
float型 10进制->2进制
![[Pasted image 20251101110642.png]]
此处如果p-len(result)<=0，不影响程序，程序不会报错，只会不输出任何内容
![[QQ2025111-11367.mp4]]![[Pasted image 20251101113805.png]]![[Pasted image 20251101113840.png]]
为什么费劲转换：That is for floats.
![[Pasted image 20251102143650.png]]
![[Pasted image 20251102143703.png]]

### APPROXIMATION
![[Pasted image 20251102144053.png]]
![[Pasted image 20251102144201.png]]
TWO SETS:
①epsilon：允许的误差范围
②增量：我们每次猜测的变化量多少（减少增量，程序会得到更精准的近似值）
`x = 54321
`epsilon = 0.01
`num_guesses = 0
`guess = 0.0
`increment = 0.0001
`while abs(guess**2 - x) >= epsilon:
`guess += increment
`num_guesses += 1
`print(f'num_guesses = {num_guesses}')
`print(f'{guess} is close to square root of {x}')
![[Pasted image 20251102144810.png]]
![[Pasted image 20251102144822.png]]
![[QQ2025112-15239.mp4]]
添加一个条件，以防上述情况的出现
`while abs(guess**2 - x) >= epsilon and guess**2 <= x:
![[Pasted image 20251102150639.png]]
![[Pasted image 20251102151150.png]]


# lecture6 bisection search
![[Pasted image 20251102161522.png]]
![[Pasted image 20251102162158.png]]

___当猜测的数字是小数时，low=x（因为sqrt（x）>x)___
![[Pasted image 20251102163723.png]]

**求负数的三次方根**
![[Pasted image 20251102172300.png]]

**Newton-Raphson**
![[Pasted image 20251102164053.png]]
![[QQ2025112-17622.mp4]]
![[Pasted image 20251102170941.png]]
guess的初始值随便选，因为算法会自动收敛

# lecture7Decomposition Abstraction and Functions
![[Pasted image 20251102174020.png]]
![[Pasted image 20251102174207.png]]
![[Pasted image 20251102184502.png]]
![[Pasted image 20251102184547.png]]

# lecture8 Functions as Objects
`def add(x,y):  
    `return x+y  
`def mult(x,y):  
    `print(x*y)   
`add(1,2)  
`print(add(2,3))  
`mult(3,4)  
`print(mult(4,5))
输出结果：
`5
`12
`20
`None  //mult没有return返回值，所以print输出None

>[!NOTE]
>函数体一旦遇到return语句，就不会继续执行接下来的程序，在循环体中也一样

独特的python函数
![[QQ2025114-14471.mp4]]
![[QQ2025114-144910.mp4]]
![[Pasted image 20251104145046.png]]
python中一个函数体可以有无数个函数名
![[QQ2025114-145315.mp4]]
函数做函数参数实例
`def apply(criteria,n):`
//此处criteria只是一个形式参数，可以是任何类型，可以是一个函数名
    `count = 0`
    `for i in range(0, n+1):`
        `if criteria(i):`
            `count += 1`
    `return count`

`def is_even(x):`
    `return x%2==0`//返回的是bool类型
`apply(is-even,10)`//调用函数参数

# lecture9 lambda functions   tuples and lists
## lambda functions
![[Pasted image 20251104151948.png]]
![[Pasted image 20251104152039.png]]
![[Pasted image 20251104152132.png]]
这两个函数是等价的
![[Pasted image 20251104152146.png]]![[Pasted image 20251104152158.png]]
当我们想要重复调用函数时，必须copy匿名函数几次，因为匿名函数没有函数名供我们调用

## tuple
![[Pasted image 20251104153355.png]]
![[Pasted image 20251104153559.png]]
![[Pasted image 20251104153530.png]]
**可以在元组中混合不同类型的数据**

### 元组的一些操作
![[Pasted image 20251104153654.png]]
如果给t重新赋值会如何？
![[QQ2025114-153840.mp4]]
>[!NOTE]
>元组可以是另一个元组的元素（不过当元组里面包含元组时，打印元组的长度，输出的长度不会受子元组影响，而是将元组元素当作长度为1来输出）



### 元组的神奇应用
1、
![[Pasted image 20251104154417.png]]
2、
![[Pasted image 20251104154509.png]]
![[Pasted image 20251104154735.png]]

3、无限参数
![[Pasted image 20251104155535.png]]
![[Pasted image 20251104155513.png]]
![[Pasted image 20251104155642.png]]
>[!NOTE]
>如果args前面没有* 号，传入参数的时候就需要传入“元组”，才能实现参数数量的可变
![[Pasted image 20251104155703.png]]

## list
![[Pasted image 20251104155818.png]]
### list一些操作
![[Pasted image 20251104155911.png]]
list遍历
![[Pasted image 20251104160455.png]]

# lecture10 lists and mutability
