"""
创建数据表
Contributor：zhangjiarui
Reviewer：xionglongfei
"""


from sqlalchemy import create_engine, Column, Integer, VARCHAR
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


engine = create_engine("mysql+pymysql://root:password@localhost/test_database", encoding='utf-8')      #本地
Base = declarative_base()

Session = sessionmaker(bind=engine)
session = Session()


class Cellphone_Data(Base):

    __tablename__ = 'Cellphone_Data'
    ID = Column(Integer(), primary_key=True , autoincrement=True, nullable=True, comment="ID")
    province = Column(VARCHAR(256), comment="省份")
    city = Column(VARCHAR(256), comment="城市")
    year = Column(VARCHAR(256), comment="年份")
    month = Column(VARCHAR(256), comment="月份")


    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.ID)

    __table_args__ = {
        "mysql_charset": "utf8"
    }


def init_db():
    Base.metadata.create_all(engine)


if __name__ == '__main__':
    init_db()
