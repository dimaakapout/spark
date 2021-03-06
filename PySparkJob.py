# python PySparkJob.py clickstream.parquet result
# spark-submit PySparkJob.py clickstream.parquet result

import io
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, when
from pyspark.sql import functions as F
from functools import reduce

def process(spark, input_file, target_path):
    # Чтение данных
    df = spark.read.parquet(input_file)

    # Список всех преобразований исходных данных
    transforms = [] 


    transforms.extend([
    
    # ad_id | integer | id рекламного объявления
    df[['ad_id']].distinct(),

    # target_audience_count | decimal | размер аудитории, на которую таргетируется объявление
    df[['ad_id', 'target_audience_count']].groupBy('ad_id').max('target_audience_count')\
        .withColumnRenamed('max(target_audience_count)', 'target_audience_count'),

    # has_video | integer | 1 если есть видео, иначе 0
    df[['ad_id', 'has_video']].groupBy('ad_id').max('has_video')\
        .withColumnRenamed('max(has_video)', 'has_video'),

    # is_cpm | integer | 1 если тип объявления CPM, иначе 0
    df.withColumn('is_cpm', when(col('ad_cost_type')=='CPM', 1).otherwise(0)).groupBy('ad_id').max('is_cpm')\
        .withColumnRenamed('max(is_cpm)', 'is_cpm'),

    # is_cpc | integer | 1 если тип объявления CPC, иначе 0
    df.withColumn('is_cpc', when(col('ad_cost_type')=='CPC', 1).otherwise(0)).groupBy('ad_id').max('is_cpc')\
        .withColumnRenamed('max(is_cpc)', 'is_cpc'),

    # ad_cost | double | стоимость объявления в рублях
    df[['ad_id', 'ad_cost']].groupBy('ad_id').max('ad_cost')\
        .withColumnRenamed('max(ad_cost)', 'ad_cost'),


    # day_count | integer | Число дней, которое показывалась реклама
    # Join'им уникальные ad_id с таблицей |ad_id|day_count|, предварительно убрав дни,
    # в которые не было просмотров (их 0)
    df[['ad_id']].distinct().join(
        df.withColumn('is_view', when(col('event')=='view', 1).otherwise(0))[['ad_id', 'date', 'is_view']]\
            .where(col('is_view')==1).groupBy('ad_id', 'date').count().groupBy('ad_id').count(), 
        'ad_id', 'left').fillna(0).withColumnRenamed('count', 'day_count')
    ])
    
    
    # CTR | double | Отношение числа кликов к числу просмотров
    # Просмотры
    views = df.withColumn('is_view', when(col('event')=='view', 1).otherwise(0)).groupBy('ad_id').sum('is_view')\
        .withColumnRenamed('sum(is_view)', 'views')
    # Клики
    clicks = df.withColumn('is_click', when(col('event')=='click', 1).otherwise(0)).groupBy('ad_id').sum('is_click')\
        .withColumnRenamed('sum(is_click)', 'clicks')
    
    # CTR (в случае, когда есть клики и нет показов, присвоено значение 0)
    transforms.append(
    views.join(clicks, 'ad_id', 'left').select(col('ad_id'), col('clicks') / col('views')).fillna(0)\
        .withColumnRenamed('(clicks / views)', 'CTR')
    )
    
    # Соединим все преобразования в единую таблицу
    join = lambda x, y: x.join(y, 'ad_id', 'left')
    ndf = reduce(join, transforms)
    
    # Train tast split
    splits = ndf.randomSplit([0.75, 0.25], 41)

    # Сохранение данных
    splits[0].coalesce(1).write.parquet(os.path.join(target_path, 'train'))
    splits[1].coalesce(1).write.parquet(os.path.join(target_path, 'test'))
    
    print('Completed!')




def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)