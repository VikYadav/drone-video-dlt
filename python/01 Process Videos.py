# Databricks notebook source
# MAGIC %pip install opencv-python 

# COMMAND ----------

import dlt

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType
import pandas as pd

IMG_OUTPUT_DIR = "dbfs:/stanford_drone/images/"
FORMAT = ".mov"

dbutils.fs.mkdirs(IMG_OUTPUT_DIR)

# COMMAND ----------

@dlt.table(
  comment="Raw videos to be processed"
)
def videos_raw():
  return (spark.read.format("binaryFile").option("pathGlobFilter", "*{}".format(".mov")).load("dbfs:/databricks-datasets-private/ML/stanford_drone/videos/").withColumn("path", regexp_replace("path", "dbfs:/", "/dbfs/")).select("path", "modificationTime", "length"))

# COMMAND ----------

@dlt.table(
  comment="Videos split into frames"
)
def split_videos():
  
  @pandas_udf(StringType())
  def extract_images(x: pd.Series) -> pd.Series:

    import cv2
    from os.path import basename

    def extract_image(y):

      count = 0
      v = cv2.VideoCapture(y)
      filename = basename(y).replace(FORMAT, "")
      success, img = v.read()
      success = True

      while success:
        v.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        success, img = v.read()
        if success:
          output_file = f"/{IMG_OUTPUT_DIR.replace(':', '')}/{filename}_frame_{count}.jpg"
          cv2.imwrite(output_file, img)
        count += 1

      return output_file

    return x.apply(extract_image)

  return (dlt.read("videos_raw").withColumn("output_path", extract_images(col("path"))))