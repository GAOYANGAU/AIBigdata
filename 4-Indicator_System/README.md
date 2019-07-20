# 数据上卷下钻展示
## 说明
该项目文件包含两部分：
 > data/  
 
 该部分为一个Django项目，通过一个html页面为用户展示数据上卷和下钻
 > data_create/
 
 该部分作用为创建Mysql库中的数据表，创建用于展示的数据，并将数据存入Mysql数据库中

--- 
## 依赖环境
* MySQL 8.0.15
* Python 3.6
* Django 2.1.8
* PyMySQL 0.9.3
* SQLAlchemy 1.3.1
---
## 执行方法
1. 在本地安装MySQL数据库，创建test_database数据库(该数据库名称在工程中三处使用，建议不修改)。
   1.1.安装mysql可通过以下脚本
   sudo apt-get install mysql-server
   1.2.配置root密码可通过(该工程代码里密码设为password)
   sudo mysql_secure_installation    
2. 执行```python create_databases.py```创建Cellphone_Data数据表。
3. 执行```python insert_data.py```创建数据，并将数据插入数据库。
4. 执行```python manage.py runserver 127.0.0.1:8000```
5. 打开浏览器，在[http://127.0.0.1:8000/echarts/](http://127.0.0.1:8000/echarts/)查看数据上卷、下钻展示。