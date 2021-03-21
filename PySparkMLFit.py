import io
import sys

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession

# Используйте как путь куда сохранить модель
MODEL_PATH = 'spark_ml_model'


def process(spark, train_data, test_data):
    #train_data - путь к файлу с данными для обучения модели
    #test_data - путь к файлу с данными для оценки качества модели
    
    # load data
    train_data = spark.read.parquet(train_data)
    test_data = spark.read.parquet(test_data)

    # Оценщик
    evaluator = RegressionEvaluator(labelCol='ctr', metricName='rmse')

    # Словарь Моделей
    key2model = {
        'lr': LinearRegression(),
        'dtree': DecisionTreeRegressor(),
        'rf': RandomForestRegressor(),
        'gbt': GBTRegressor()
    }
    
    # Словарь параметров моделей
    model2param = {
        'lr': ParamGridBuilder()\
        .addGrid(key2model['lr'].labelCol, ['ctr']) \
        .addGrid(key2model['lr'].maxIter, [40, 75, 100]) \
        .addGrid(key2model['lr'].regParam, [.0, .2, .5]) \
        .addGrid(key2model['lr'].elasticNetParam, [.0, .2, .8]) \
        .build(),
        'dtree': ParamGridBuilder()\
        .addGrid(key2model['dtree'].labelCol, ['ctr']) \
        .addGrid(key2model['dtree'].maxDepth, [3, 4, 5]) \
        .addGrid(key2model['dtree'].maxBins, [32, 64]) \
        .build(),
        'rf': ParamGridBuilder()\
        .addGrid(key2model['rf'].labelCol, ['ctr']) \
        .addGrid(key2model['rf'].maxDepth, [3, 5, 7]) \
        .addGrid(key2model['rf'].numTrees, [5, 15, 25, 40]) \
        .build(),
        'gbt': ParamGridBuilder()\
        .addGrid(key2model['gbt'].labelCol, ['ctr']) \
        .addGrid(key2model['gbt'].maxDepth, [3, 5, 7]) \
        .addGrid(key2model['gbt'].maxIter, [20,30,40]) \
        .addGrid(key2model['gbt'].lossType, ['squared', 'absolute']) \
        .build()
    }

    feature = VectorAssembler(inputCols=train_data.columns[:-1],outputCol="features") # Preprocessing
    models = dict() # Словарь всех обучаемых моделей
    tvs_dict = dict() # TrainValidationSplit objects, оптимизируемые модели

    for key in key2model.keys():
        models[key] = Pipeline(stages=[feature, key2model[key]]) # Создаем Pipeline для каждой модели
        tvs_dict[key] = TrainValidationSplit(estimator=models[key],
                               estimatorParamMaps=model2param[key],
                               evaluator=RegressionEvaluator(labelCol='ctr', metricName='rmse'),
                               trainRatio=0.8)
        print(key)
        print('Обучение {0} ...'.format(key))
        tvs_dict[key] = tvs_dict[key].fit(train_data)
        tvs_dict[key].bestModel.write().overwrite().save(key + '_model')
        print('Обучение {0}. Модель сохранена'.format(key))
        predict = tvs_dict[key].bestModel.transform(test_data)
        score = evaluator.evaluate(predict)
        print('RMSE(test) = {0}'.format(score))

def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)