# 0 推荐系统概述

    RecommendEngine新闻推荐引擎，有两个流程构成：
    1 粗排：根据所有用户的点击行为，采用Item CF算法召回新闻，构成召回候选集。
    2 精排：根据用户的点击行为，采用LDA的主题分布的余弦相似度对召回候选集做精排，精排的结果推送给用户
    输入：用户ID
    输出：文章ID

# 1 recommend.py 推荐新闻主类
    推荐系统的主模块，由基于ITEM_CF实现的召回模块和基于LDA主题分布相似度实现的排序模块构成
    运行脚本: python recommend.py,完成一次新闻推荐

# 2 config.py:配置文件
    msyql配置参数
# 3 resource.py: 资源文件
    模型文件，mysql实例
# 4 item_cf.py:基于物品的协同过滤
    实现基于物品的协同过滤
# 5 news_sim_lda.py:基于LDA的 排序模型
    实现基于LDA主题分布的文章相似度计算
# 6 mysql_dao.py:mysql工具类
        封装mysql操作的工具类
# 7 train_lda_model.py 训练LDA模型
    离线训练LDA模型

# 8 recsys.sql 存放新闻推荐系统所有数据的数据库导出文件

    读者可利用此文件导入自己的mysql数据库


# 9 stopwords.txt
    停用词词表

# 10 msyql表结构：
    推荐系统数据由两个mysql表组成：存放新闻数据的表、存放用户点击行为


## 10.1 存放新闻的msyql表
### 表结构

        +---------+------------------+------+-----+---------+-------+
        | Field   | Type             | Null | Key | Default | Extra |
        +---------+------------------+------+-----+---------+-------+
        | newsid  | int(10) unsigned | NO   | PRI | NULL    |       |
        | title   | varchar(100)     | NO   | MUL | NULL    |       |
        | label   | varchar(20)      | NO   | MUL | NULL    |       |
        | content | longtext         | YES  |     | NULL    |       |
        +---------+------------------+------+-----+---------+-------+


### 建表语句

        CREATE TABLE `article` (
        `newsid` int(10) unsigned NOT NULL,
        `title` varchar(100) NOT NULL,
        `label` varchar(20) NOT NULL,
        `content` longtext,
           PRIMARY KEY (`newsid`),
        KEY `title` (`title`),
        KEY `label` (`label`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8


## 10.2 存放用户行为的msyql表
### 表结构

        +------------+------------------+------+-----+---------+----------------+
        | Field      | Type             | Null | Key | Default | Extra          |
        +------------+------------------+------+-----+---------+----------------+
        | id         | int(11)          | NO   | PRI | NULL    | auto_increment |
        | userid     | int(10) unsigned | NO   | MUL | NULL    |                |
        | newsid     | int(10) unsigned | NO   | MUL | NULL    |                |
        | click_date | varchar(10)      | NO   | MUL | NULL    |                |
        +------------+------------------+------+-----+---------+----------------+

### 建表语句

           CREATE TABLE `user_click` (
          `id` int(11) NOT NULL AUTO_INCREMENT,
          `userid` int(10) unsigned NOT NULL,
          `newsid` int(10) unsigned NOT NULL,
          `click_date` varchar(10) NOT NULL,
          PRIMARY KEY (`id`),
          KEY `userid` (`userid`),
          KEY `newsid` (`newsid`),
          KEY `click_date` (`click_date`)
        ) ENGINE=InnoDB AUTO_INCREMENT=1014 DEFAULT CHARSET=utf8


# 安装和运行 Tips:
## 1.安装mysql可通过以下脚本
sudo apt-get install mysql-server
## 2.配置root密码可通过
sudo mysql_secure_installation
## 3.创建数据库(可创建git仓库里同名的数据库)
create database recsys
## 4.导入数据库(git仓库里已创建好了相应的数据库和表以及数据，可直接导入使用)
mysql -u root -p recsys < recsys.sql
## 5.直接运行recommend.py 查看当前数据下的推荐结果，可修改代码里__main__的userid查看给其它用户的推荐结果。亦可修改数据库表里的内容或重新录入新的表单来体验不同的推荐结果。

