
# ✅ MySQL **学到什么程度**就可以进入下一步（JDBC / SpringBoot）？

**不是学到“很熟”才继续，而是学到“能支持你的学生管理系统”就可以继续。**

也就是说，你只需要掌握：

### 1）**基础 SQL（CRUD）**

能写出：

```sql
SELECT * FROM student;
INSERT INTO student(name, age) VALUES('Alice', 19);
UPDATE student SET age = 20 WHERE id = 3;
DELETE FROM student WHERE id = 5;
```

### 2）**表结构建表**

能根据你的 Student 类建表，比如：

```sql
CREATE TABLE student (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50),
    age INT,
    gender VARCHAR(10)
);
```

### 3）**简单条件查询**

比如：

```sql
SELECT * FROM student WHERE age > 18;
```

如果你能做到这三件事：

> **你已经可以开始下一阶段（JDBC）了。**

---

## ⏱ 大概需要多长时间？

|内容|所需时间|标准|
|---|---|---|
|增删改查（CRUD）|**1~2 天**|能手写几条不看书|
|建表 & 数据类型|**1 天**|能为你的实体类设计表|
|条件查询 + 模糊查询|**0.5 天**|能查符合条件的数据|

**总计：3~4 天** 足够 ✅

你不需要：

- 学触发器
    
- 学存储过程
    
- 学事务隔离级别
    
- 学索引优化
    

这些是 **以后面试再学**，不是现在要学的内容。

---

## 🎯 学习的判断标准（自测）

如果你能回答下面的问题，你就可以进入下一步了：

> 你能否把你写的 `Student` 类 1:1 在数据库中建表？  
> 你能否手动往表中插入几条学生数据？  
> 你能否根据姓名或 ID 查到学生信息？

如果答案是 **可以** → **开始 JDBC**

如果 **不可以** → 说明还没有学会最基础的东西，不要急。

---

## ✅ 下一步是什么？

**JDBC → 用 Java 操作你建好的数据库**

它是关键的一步，让你实现：

```
Java 程序 <——> MySQL 数据库
```

从这一步开始，你的项目就会从“假“变成“真的”。

---

## ⭐ 总结一句话

> **MySQL 学到可以支持你学生管理系统的数据存储 → 就立刻进入 JDBC，不要等学很久再开始。**

---



java：首先黑马二倍速Java入门上册
看书（日常看java卷1主要看类这一块，cpp pirmer plus先暂停（有问题再看），数据结构啃书（当作字典，有问题再看，主要看代码随想录）
网站+项目：当前可做：代码随想录的项目
后续：苍穹外卖+代码随想录知识星球找高级项目来做



---

# 学习路线
## ❌ 你原来的路线

```
Java → JavaWeb → SSSM → MyBatis-Plus → RabbitMQ → Redis
```

问题：

- **JavaWeb太老**（Servlet + JSP），企业现在基本不用
    
- **SSSM（Spring + SpringMVC + MyBatis）** 也不再手动搭了（现在都是 SpringBoot 自动集成）
    
- 顺序缺乏项目支撑，东西堆得太快
    

---

## ✅ 我给你改的 **正确就业路线（2025 企业上岗版）**

### **阶段 1：语言基础**

```
Java 基础（语法、面向对象、集合、异常、I/O）
```

目标：能写 **学生管理系统（集合版）**

---

### **阶段 2：数据库基础**

```
MySQL（CRUD + 建表 + 条件查询）
JDBC（用 Java 操作数据库）
```

目标：学生管理系统升级为 **数据库版**

---

### **阶段 3：现代后端核心**

```
SpringBoot ← 代替 JavaWeb + Spring + SpringMVC
MyBatis / MyBatis-Plus
```

目标：能写完整的 Web 后端接口 + 前端可调用

> 解释下：  
> **SpringBoot** = 核心框架  
> **MyBatis** = Java ↔ 数据库 之间的桥  
> **MyBatis-Plus** = MyBatis 的增强版（更省代码）

也就是说：  
**SSM = 过时的搭法**  
**SpringBoot + MyBatis-Plus = 现在的主流搭法**

---

### **阶段 4：中级能力（提升竞争力）**

```
Redis （缓存 · 分布式锁）
RabbitMQ 或 Kafka（消息队列）
```

目标：做项目时能解决 “高并发 & 系统解耦” 问题  
这是你拉开差距、提高薪资的地方。

---

## 🔥 最终你会写出的项目形态

```
SpringBoot + MyBatis-Plus + MySQL + Redis + MQ
```

这是 **企业原汁原味的后端技术栈**。  
你完成这一套 → **本科直就完全可行**。

---

## ⭐ 最终学习路径总结（可直接照执行）

```
Java 基础
→ 数据结构（同步进行）
→ 学生管理系统（集合版）
→ MySQL + JDBC
→ 学生管理系统（数据库版）
→ SpringBoot
→ MyBatis + MyBatis-Plus
→ 做真实业务项目（代码随想录知识星球）
→ Redis（缓存 + 分布式锁）
→ RabbitMQ（异步消息）
→ 项目打磨 + 面试准备
```

---

## 🎯 最重要的一句话

> **不要在没有项目的时候学 Redis / MQ。  
> 一定是项目需要时再引入。**

否则你学了也不知道能干嘛，记不住，容易放弃。

---
## 🎯 阶段结束与否的 **判断标准**

如果你能做到：

`启动程序 → 控制台显示菜单 1. 添加学生 2. 删除学生 3. 修改学生信息 4. 查看学生列表`

并且全部基于 **ArrayList** 实现，  
那么你可以 **直接进入下一阶段：MySQL**。

#  MyBatis 学到这个程度就可以进入 MyBatis-Plus / SpringBoot 项目**

## **1）能够独立完成基本 CRUD 映射（核心）**

你要能写：

`<select id="selectById" parameterType="int" resultType="Student">     SELECT * FROM student WHERE id = #{id} </select>  <insert id="insertStudent" parameterType="Student">     INSERT INTO student(name, age) VALUES(#{name}, #{age}) </insert>  <update id="updateStudent" parameterType="Student">     UPDATE student SET name=#{name} WHERE id=#{id} </update>  <delete id="deleteStudent" parameterType="int">     DELETE FROM student WHERE id = #{id} </delete>`

**会写就行，不要求背。**

---

## **2）能够使用动态 SQL（最重要的技能点）**

能看懂并会写：

`<select id="selectByCondition" resultType="Student">     SELECT * FROM student     <where>         <if test="name != null">             AND name LIKE CONCAT('%',#{name},'%')         </if>         <if test="age != null">             AND age = #{age}         </if>     </where> </select>`

这个就是你之后 **做搜索 / 筛选 / 后台查询** 的核心能力。

---

## **3）会做一对多 / 多对一联合查询（企业常用）**

比如：

`<select id="selectStudentWithCourses" resultMap="studentCourseMap">     SELECT s.id, s.name, c.id AS courseId, c.courseName     FROM student s     LEFT JOIN course c ON s.id = c.studentId </select>`

但是 **不需要深扒 ORM 原理**，那是你实习/面试再补。

---

## **4）学会 XML + 注解二选一即可**

推荐：

| 新手阶段 | 用 XML（可视性强） |  
| 进项目阶段 | 用 MyBatis-Plus 注解/Wrapper |

你不需要同时精通两套。

---

## ⏱ **时间分配方案（20 小时完全够）**

|内容|时间|目标|
|---|---|---|
|MyBatis 基本概念、项目搭建|3h|知道它在项目中负责什么|
|CRUD + resultType + parameterType|4h|能写基础数据接口|
|**动态 SQL（核心）**|**6h**|会写带条件的查询|
|多表查询（join + resultMap）|4h|能处理 80% 实际业务|
|整合 SpringBoot|3h|真正能跑在 Web 项目里|

**总计：20 小时 → 就能直接进 MyBatis-Plus 阶段。**

# 1 Java开篇
## 1.1cmd
### 1.1.1常见cmd命令
![[Pasted image 20251113131157.png]]
## 1.2环境配置
![[Pasted image 20251113133423.png]]

# 2 Java基础
## 2.1变量
### 2.1.1变量的注意事项
![[Pasted image 20251113163316.png]]
### 2.1.2数据类型
![[Pasted image 20251113170838.png]]
![[Pasted image 20251113171037.png]]
![[Pasted image 20251113171053.png]]

### 2.1.3标识符
标识符：给类，方法，变量等起的名字

1、硬性规定
①由数字、字母、下划线和美元符组成
②不能以数字开头
③不能是关键字
④区分大小写

2、软性规定（方法，变量）
①标识符是一个单词的时候，全部小写
②标识符由多个单词组成的时候，第一个单词首字母小写，其他单词首字母大写

3、软性规定（类名）
①标识符是一个单词的时候，首字母大写
②标识符由多个单词组成的时候，每个单词的首字母大写

### 2.1.4键盘录入
![[Pasted image 20251113171903.png]]


## 2.2计算机中的数据存储
### 2.2.1进制
![[Pasted image 20251113164838.png]]

![[Pasted image 20251113165417.png]]
![[Pasted image 20251113165557.png]]

### 2.2.2计算机的三原色
![[Pasted image 20251113170458.png]]

## 2.3运算符
### 2.3.1算术运算符
>[!attention]
>除法：
整数参与计算，结果只有整数
小数参与计算，结果有可能不精确

将整数拆分成个位 十位 百位 ……
![[Pasted image 20251113182303.png]]
#### 1、类型转换
![[Pasted image 20251113182719.png]]
①隐式转换
![[Pasted image 20251113182806.png]]
![[Pasted image 20251113182844.png]]
②强制转换
![[Pasted image 20251113183147.png]]
![[Pasted image 20251113183259.png]]
>[!attention]
>byte类型会先转换为int，所以为byte result赋值时，要进行强制转换

#### 2、字符串+操作
![[Pasted image 20251113183442.png]]
**只要出现字符串就会开始拼接，不管后面的数据类型，如果后面有其他+号，一律视为字符串拼接**
![[Pasted image 20251113183541.png]]
![[Pasted image 20251113183835.png]]

#### 3、字符+操作

