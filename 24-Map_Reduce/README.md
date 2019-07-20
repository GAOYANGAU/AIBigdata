本地测试：
cat test.txt |./mapper.py | sort |./reducer.py
Tips：未加执行权限时可通过cat test.txt |python mapper.py |sort | python reducer.py 命令运行，其中python为指定的解释器

hadoop 测试：
路径：
mapper.py : /home/hadoop/mapreduce/项目/mapper.py
reducer.py: /home/hadoop/mapreduce/项目/reducer.py
hdfs: /hdfsdata/test.txt

运行：
hadoop jar /data/app/hadoop-3.1.1/share/hadoop/tools/lib/hadoop-streaming-3.1.1.jar \
-file /home/hadoop/mapreduce/1_workcount/mapper.py -mapper /home/hadoop/mapreduce/1_workcount/mapper.py \
-file /home/hadoop/mapreduce/1_workcount/reducer.py -reducer /home/hadoop/mapreduce/1_workcount/reducer.py \
-input /hdfsdata/test.txt -output /hdfsdata/output