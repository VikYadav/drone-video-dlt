# Databricks notebook source
# MAGIC %pip install opencv-python

# COMMAND ----------

dbutils.widgets.text("INPUT_DIR", defaultValue="dbfs:/aweaver/video_processing/videos/")
dbutils.widgets.text("OUTPUT_DIR", defaultValue="dbfs:/aweaver/video_processing/images/")
dbutils.widgets.dropdown("OUTPUT_FORMAT", defaultValue=".jpg", choices=[".jpg"])
dbutils.widgets.dropdown("VIDEO_EXTENSION", defaultValue=".mp4", choices=[".mp4", ".mov"])

INPUT_DIR = dbutils.widgets.get("INPUT_DIR")
OUTPUT_DIR = dbutils.widgets.get("OUTPUT_DIR")
OUTPUT_FORMAT = dbutils.widgets.get("OUTPUT_FORMAT")

dbutils.fs.mkdirs(f"{OUTPUT_DIR}/videos/")
dbutils.fs.mkdirs(f"{OUTPUT_DIR}/frames/")

spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

# COMMAND ----------

from pyspark.sql.functions import sha2

split_videos =  spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*{}".format(VIDEO_EXTENSION)) \
    .option("recursiveFileLookup", "true") \
    .load(INPUT_DIR) \
    .withColumn("content_hash", sha2("content", 512))

display(split_videos)

# COMMAND ----------

from pyspark.sql.types import *
import pandas as pd

schema = StructType([
  StructField("path", StringType(), False),
  StructField("modificationTime", TimestampType(), False),
  StructField("length", LongType(), False),
  StructField("content", BinaryType(), False),
  StructField("content_hash", StringType(), False),
  StructField("output_paths", StringType(), False)
])

def extract_frames(pdf: pd.DataFrame) -> pd.DataFrame:

  import cv2
  from os.path import basename

  video_file = f"/{pdf['path'].get(0).replace(':', '')}"
  v = cv2.VideoCapture(video_file)
  base_filename = basename(video_path).replace(VIDEO_EXTENSION, "")

  if(v.isOpened() == False): 
    raise Exception("Cannot open {}".format(video_path))
     
  count = 0
  images = []
  
  while(v.isOpened()):

    success, frame = v.read()
    
    output_filename = f"/{OUTPUT_DIR.replace(':', '')}/{base_filename}_frame_{count}.jpg"
    local_file = f"/tmp/{output_filename}"

    if success:
      
      # Check whether the target file exists or not before creating it
      if not os.path.isfile(output_filename):
        cv2.imwrite(local_file, frame)
        
      try:
        import shutil
        # copy-then-delete, because it's moving between different filesystems, 
        # and writing locally before copying to an NFS mount is safer
        shutil.move(local_file, output_filename)
        images.append(f"{OUTPUT_DIR}/{base_filename}_frame_{count}.jpg")
      
    except Exception as e:
      print(f"Error moving {local_file} to {video_file} : {e}")\
      
    else: 
      break
    count += 1
   
  v.release()
  cv2.destroyAllWindows()
  
  # merge the output files back together with the input files...
  images = pd.DataFrame({"output_paths": images, "path": pdf['path'].get(0)})

  return images.merge(pdf, how="left", on="path")

display(split_videos.groupby("path").applyInPandas(extract_frames, schema=schema))

# COMMAND ----------


