# Databricks notebook source
import pandas as pd
from pyspark.context import SparkContext
from pyspark.sql import HiveContext
from textblob import TextBlob

# COMMAND ----------

reviews = spark.read.format('csv').options(header='true',inferSchema='true',\
                                           quote='"',delimiter=',',\
                                           escape='"',quoteMode=None, \
                                           enforceSchema=True)\
                                  .load('/FileStore/tables/reviews')
display(reviews)

# COMMAND ----------

reviews.printSchema()

# COMMAND ----------

reviews.groupBy('company').count().show()

# COMMAND ----------

reviews.describe('overall-ratings').show()

# COMMAND ----------

reviews.filter(reviews.company=='google').describe('overall-ratings').show()
reviews.filter(reviews.company=='amazon').describe('overall-ratings').show()
reviews.filter(reviews.company=='microsoft').describe('overall-ratings').show()
reviews.filter(reviews.company=='facebook').describe('overall-ratings').show()
reviews.filter(reviews.company=='netflix').describe('overall-ratings').show()
reviews.filter(reviews.company=='apple').describe('overall-ratings').show()

# COMMAND ----------

print(reviews.select('location').distinct().count())
print(reviews.select('job-title').distinct().count())

# COMMAND ----------

reviews.describe(['overall-ratings', 'work-balance-stars', 'culture-values-stars',\
                 'carrer-opportunities-stars', 'comp-benefit-stars', 'senior-mangemnet-stars',\
                 'helpful-count']).show()

# COMMAND ----------

# Fill missing data with average value
reviews = reviews.replace('none', None).fillna({'work-balance-stars':'3.37325448',\
                                               'culture-values-stars':'3.784450',\
                                               'carrer-opportunities-stars':'3.6340345',\
                                               'comp-benefit-stars':'3.94250265',\
                                               'senior-mangemnet-stars':'3.322522',
                                               'location':'N/A', 'pros':'', 'cons':'',\
                                               'advice-to-mgmt':'', 'summary':'',\
                                               'job-title':''})

# COMMAND ----------

# Convert to double type
reviews = reviews.withColumn('work-balance-stars', reviews['work-balance-stars'].cast('double'))
reviews = reviews.withColumn('culture-values-stars', reviews['culture-values-stars'].cast('double'))
reviews = reviews.withColumn('carrer-opportunities-stars', reviews['carrer-opportunities-stars'].cast('double'))
reviews = reviews.withColumn('comp-benefit-stars', reviews['comp-benefit-stars'].cast('double'))
reviews = reviews.withColumn('senior-mangemnet-stars', reviews['senior-mangemnet-stars'].cast('double'))

# COMMAND ----------

@udf
def get_length(cell):return len(cell.split('.'))

# COMMAND ----------

from pyspark.sql.functions import lit
reviews = reviews.withColumn('summary_length', lit(get_length(reviews['summary'])))
reviews = reviews.withColumn('summary_length', reviews['summary_length'].cast('int'))
reviews.select('summary_length').describe().show()

# COMMAND ----------

reviews = reviews.withColumn('pros_length', lit(get_length(reviews['pros'])))
reviews = reviews.withColumn('pros_length', reviews['pros_length'].cast('int'))
reviews.select('pros_length').describe().show()

# COMMAND ----------

reviews = reviews.withColumn('cons_length', lit(get_length(reviews['cons'])))
reviews = reviews.withColumn('cons_length', reviews['cons_length'].cast('int'))
reviews.select('cons_length').describe().show()

# COMMAND ----------

# Build target variable
from pyspark.sql.functions import col, expr, when
from pyspark.sql.functions import avg
from pyspark.sql.window import Window
import sys

w = Window.partitionBy('company').rangeBetween(-sys.maxsize,sys.maxsize)
reviews = reviews.withColumn('average',
                             avg(reviews['overall-ratings']).over(w))\
                 .withColumn('labels',
                             when((col('average')>col('overall-ratings')), 0).otherwise(1))
reviews.groupby('labels').count().show()

# COMMAND ----------

@udf
def get_sentiment(cell):
  return TextBlob(cell).sentiment.polarity

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT
to_vector = udf(lambda vs: Vectors.dense([float(i) for i in vs]), VectorUDT())

# COMMAND ----------

reviews = reviews.withColumn('summary_sentiment_avg', lit(get_sentiment(reviews['summary'])))
reviews = reviews.withColumn('pros_sentiment_avg', lit(get_sentiment(reviews['pros'])))
reviews = reviews.withColumn('cons_sentiment_avg', lit(get_sentiment(reviews['cons'])))

# COMMAND ----------

reviews = reviews.withColumn('summary_sentiment_avg', reviews.summary_sentiment_avg.cast('double'))
reviews = reviews.withColumn('pros_sentiment_avg', reviews.pros_sentiment_avg.cast('double'))
reviews = reviews.withColumn('cons_sentiment_avg', reviews.cons_sentiment_avg.cast('double'))

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer, StringIndexer, StopWordsRemover, IDF
from pyspark.ml.feature import VectorAssembler, StandardScaler, ChiSqSelector
from pyspark.ml.linalg import Vectors

sw = StopWordsRemover().getStopWords()

summary_tokenizer = Tokenizer(inputCol="summary", outputCol="summary_words")
summary_remover = StopWordsRemover(inputCol="summary_words", outputCol="summary_filtered",
                                   stopWords=sw)
summary_hashingTF = HashingTF(inputCol='summary_filtered', outputCol="summary_raw",
                              numFeatures=500)
summary_idf = IDF(inputCol="summary_raw", outputCol="summary_features")

pros_tokenizer = Tokenizer(inputCol="pros", outputCol="pros_words")
pros_remover = StopWordsRemover(inputCol="pros_words", outputCol="pros_filtered",
                              stopWords=sw)
pros_hashingTF = HashingTF(inputCol='pros_filtered', outputCol="pros_raw",
                          numFeatures=500)
pros_idf = IDF(inputCol="pros_raw", outputCol="pros_features")

cons_tokenizer = Tokenizer(inputCol="cons", outputCol="cons_words")
cons_remover = StopWordsRemover(inputCol="cons_words", outputCol="cons_filtered",
                               stopWords=sw)
cons_hashingTF = HashingTF(inputCol='cons_filtered', outputCol="cons_raw",
                           numFeatures=500)
cons_idf = IDF(inputCol="cons_raw", outputCol="cons_features")

advice_tokenizer = Tokenizer(inputCol="advice-to-mgmt", outputCol="advice_words")
advice_remover = StopWordsRemover(inputCol="advice_words", outputCol="advice_filtered",
                                 stopWords=sw)
advice_hashingTF = HashingTF(inputCol='advice_filtered', outputCol="advice_raw",
                             numFeatures=500)
advice_idf = IDF(inputCol="advice_raw", outputCol="advice_features")

title_tokenizer = Tokenizer(inputCol='job-title', outputCol='job_title_words')
title_remover = StopWordsRemover(inputCol="job_title_words", outputCol='title_filtered',
                                 stopWords=sw)
title_hashingTF = HashingTF(numFeatures=100, inputCol='title_filtered', outputCol='title_raw')
title_idf = IDF(inputCol="title_raw", outputCol="title_features")

indexer_loc = StringIndexer(inputCol="location", outputCol="location_features")

assembler1 = VectorAssembler(
    inputCols = ["location_features", "summary_features", "cons_features", "pros_features",
                 "advice_features", "helpful-count", 'title_features'],
    outputCol = "feature_baseline")
assembler2 = VectorAssembler(
    inputCols = ['senior-mangemnet-stars',"location_features", "helpful-count",
                 'cons_sentiment_avg', 'summary_sentiment_avg', 'pros_sentiment_avg',
                'work-balance-stars', 'work-balance-stars', 'culture-values-stars',
                'carrer-opportunities-stars', 'comp-benefit-stars'],
    outputCol = "feature_reduced")

feature_baseline_scaler = StandardScaler(inputCol="feature_baseline", outputCol="scaledFeatures1",
                        withStd=True, withMean=False)
feature_reduced_scaler = StandardScaler(inputCol="feature_reduced", outputCol="scaledFeatures2",
                        withStd=True, withMean=False)

pipeline = Pipeline(stages=[summary_tokenizer, summary_remover, summary_hashingTF, summary_idf, pros_tokenizer,
                            pros_remover, pros_hashingTF, pros_idf, cons_tokenizer, cons_remover, cons_hashingTF,cons_idf,
                            indexer_loc, advice_tokenizer, advice_remover, advice_hashingTF, advice_idf,
                            title_tokenizer, title_remover, title_hashingTF, title_idf, assembler1, assembler2,
                            feature_baseline_scaler, feature_reduced_scaler])

reviews = pipeline.fit(reviews).transform(reviews)
