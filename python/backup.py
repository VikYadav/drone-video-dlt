# Databricks notebook source
# MAGIC %pip install python-opencv

# COMMAND ----------

dbutils.widgets.text("INPUT_DIR", defaultValue="dbfs:/databricks-datasets-private/ML/stanford_drone/")
dbutils.widgets.text("OUTPUT_DIR", defaultValue="dbfs:/aweaver/video_processing/")
dbutils.widgets.dropdown("VIDEO_SPLIT_DURATION", defaultValue="30", choices=["10", "20", "30", "60", "120"])
dbutils.widgets.dropdown("VIDEO_EXTENSION", defaultValue=".mp4", choices=[".mp4", ".mov"])

INPUT_DIR = dbutils.widgets.get("INPUT_DIR")
OUTPUT_DIR = dbutils.widgets.get("OUTPUT_DIR")
VIDEO_SPLIT_DURATION = int(dbutils.widgets.get("VIDEO_SPLIT_DURATION"))
VIDEO_EXTENSION = dbutils.widgets.get("VIDEO_EXTENSION")

dbutils.fs.mkdirs(f"{OUTPUT_DIR}/videos/")
dbutils.fs.mkdirs(f"{OUTPUT_DIR}/frames/")

# COMMAND ----------

raw_videos =  spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*{}".format(VIDEO_EXTENSION)) \
    .option("recursiveFileLookup", "true") \
    .load(INPUT_DIR) \
    .select("path", "modificationTime", "length") # the files we're processing at this stage are too large to read the content in...

display(raw_videos)

# COMMAND ----------

import pandas as pd

from pyspark.sql.types import *

schema = StructType([
  StructField("path", StringType(), False),
  StructField("modificationTime", TimestampType(), False),
  StructField("length", LongType(), False),
  StructField("duration_in_secs", DoubleType(), False),
  StructField("output_paths", StringType(), False)
])

def split_video(pdf: pd.DataFrame) -> pd.DataFrame:
  
  import os
  from os.path import basename
  from moviepy.editor import VideoFileClip
  from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
  
  video_file = f"/{pdf['path'].get(0).replace(':', '')}"
  
  clip = VideoFileClip(video_file)
  duration = clip.duration
  pdf["duration_in_secs"] = duration
  
  extracted_files = []

  for i in range(0, round(duration + 0.5), VIDEO_SPLIT_DURATION):   
    
    start, end = i, i + VIDEO_SPLIT_DURATION
    
    if end > duration:
      end = round(duration + 0.5)
    
    output_filename = f"{basename(video_file.replace(VIDEO_EXTENSION, ''))}_{start}s_to_{end}s{VIDEO_EXTENSION}"
    local_file = f"/tmp/{output_filename}"
    
    if not os.path.isfile(output_filename):
      
      # Check whether the target file exists or not before creating it
      ffmpeg_extract_subclip(video_file, start, end, local_file) 
      output_path = f"/{OUTPUT_DIR}videos/".replace(":", "") + output_filename
    
    try:
      import shutil
      # copy-then-delete, because it's moving between different filesystems, 
      # and writing locally before copying to an NFS mount is safer
      shutil.move(local_file, output_path)
      extracted_files.append(output_path.replace("/dbfs/", "dbfs:/"))
      
    except Exception as e:
      print(f"Error moving {local_file} to {video_file} : {e}")
    
    # merge the output files back together with the input files...
    files = pd.DataFrame({"output_paths": extracted_files, "path": pdf['path'].get(0)})
  
  return files.merge(pdf, how="left", on="path")

display(raw_videos.groupby("path").applyInPandas(split_video, schema=schema))

# COMMAND ----------

split_videos =  spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*{}".format(VIDEO_EXTENSION)) \
    .option("recursiveFileLookup", "true") \
    .load(f"{OUTPUT_DIR}videos") 

display(split_videos)

# COMMAND ----------

@pandas_udf(ArrayType(StringType()))
def extract_images(video: pd.Series) -> pd.Series:
  
  import cv2
  from os.path import basename

  video_path = f"/{video.get(0)}".replace(":", "")
  v = cv2.VideoCapture(video_path)
  base_filename = basename(video_path).replace(FORMAT, "")

  if(v.isOpened() == False): 
    raise Exception("Cannot open {}".format(video_path))
     
  count = 0
  extracted_frames = []
  while(v.isOpened()):

    success, frame = v.read()

    if success:
      output_file = f"/{IMG_OUTPUT_DIR.replace(':', '')}/{base_filename}_frame_{count}.jpg"
      cv2.imwrite(output_file, frame)
      extracted_frames.append(output_file)
    else: 
      break
    count += 1
   
  v.release()
  cv2.destroyAllWindows()
  
  return pd.Series(extracted_frames)

# COMMAND ----------

bronze = raw_videos.withColumn("img_path", extract_images("path"))
display(bronze)

# COMMAND ----------

dbutils.fs.ls(IMG_OUTPUT_DIR)

# COMMAND ----------

def extract_frames(pdf: pd.DataFrame) -> pd.DataFrame:

  import cv2
  from os.path import basename

  video_path = f"/{pdf['path'][0]}".replace(":", "")
  v = cv2.VideoCapture(video_path)
  base_filename = basename(video_path).replace(FORMAT, "")

  if(v.isOpened() == False): 
    raise Exception("Cannot open {}".format(video_path))
     
  count = 0
  images = []
  while(v.isOpened()):

    success, frame = v.read()

    if success:
      output_file = f"/{IMG_OUTPUT_DIR.replace(':', '')}/{base_filename}_frame_{count}.jpg"
      cv2.imwrite(output_file, frame)
      #images.append(f"{IMG_OUTPUT_DIR}/{base_filename}_frame_{count}.jpg")
    else: 
      break
    count += 1
   
  #pdf.insert(len(pdf.columns()), "img_path", images)
  v.release()
  cv2.destroyAllWindows()

  return pdf

# COMMAND ----------

df = spark.readStream.format("cloudFiles") \
  .option("cloudFiles.format", "binaryFile") \
  .load(IMG_OUTPUT_DIR) 
  .writeStream \
  .option("checkpointLocation", "<path_to_checkpoint>") \
  .start("<path_to_target")
display(df)

# COMMAND ----------

def extract_frames(pdf: pd.DataFrame) -> pd.DataFrame:

  import cv2
  from os.path import basename

  video_path = f"/{pdf['path'][0]}".replace(":", "")
  v = cv2.VideoCapture(video_path)
  base_filename = basename(video_path).replace(FORMAT, "")

  if(v.isOpened() == False): 
    raise Exception("Cannot open {}".format(video_path))
     
  count = 0
  images = []
  while(v.isOpened()):

    success, frame = v.read()

    if success:
      output_file = f"/{IMG_OUTPUT_DIR.replace(':', '')}/{base_filename}_frame_{count}.jpg"
      cv2.imwrite(output_file, frame)
      images.append(f"{IMG_OUTPUT_DIR}/{base_filename}_frame_{count}.jpg")
    else: 
      break
    count += 1
   
  pdf.insert(len(pdf.columns()), "img_path", images)
  v.release()
  cv2.destroyAllWindows()

  return pdf

# COMMAND ----------

raw_videos =  spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*{}".format(".mov")) \
    .option("recursiveFileLookup", "true") \
    .load("dbfs:/databricks-datasets-private/ML/stanford_drone/") \
    .select("path", "modificationTime", "length")

display(raw_videos)

# COMMAND ----------

@dlt.table(
  name="bronze",
  comment="Metadata relating to raw videos to be processed"
)
@dlt.expect_all({"valid_length": "length > 0", "valid_generated_fields": "local_path IS NOT NULL"})
def videos_raw():
  return (
    spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*{}".format(FORMAT)) \
    .load("dbfs:/databricks-datasets-private/ML/stanford_drone/videos/") \
    .withColumn("local_path", regexp_replace("path", "dbfs:/", "/dbfs/")) \
    .select("path", "local_path", "modificationTime", "length")) 

# COMMAND ----------

@dlt.table(
   name="silver",
   comment="Metadata relating to videos split into image frames"  
)
@dlt.expect_all({"valid_length": "length > 0", "valid_generated_fields": "local_path IS NOT NULL"})
def extract_images():

  return (dlt.read("bronze").withColumn("img_path", extract_frames(col("local_path"))))

# COMMAND ----------

@dlt.table(
  name="gold",
  comment="Image frames loaded into a Delta Lake"
)
@dlt.expect_all({"valid_length": "length > 0", "has_content": "content IS NOT NULL", "valid_generated_fields": "content_hash IS NOT NULL"})
def load_images():
  return (
  dlt.read("silver").select("img_path").alias("path").join(spark.readStream.format("cloudFiles").option("cloudFiles.format", "binaryFile").option("recursiveFileLookup", "true") \
  .option("pathGlobFilter", "*.jpg") \
  .load(IMG_OUTPUT_DIR).withColumn("content_hash", sha2("content", 512)).join(dlt.read("silver"), "path")))

# COMMAND ----------

vids = spark.readStream.format("cloudFiles") \
  .option("cloudFiles.format", "binaryFile") \
  .option("recursiveFileLookup", "true") \
  .option("pathGlobFilter", "*.mov") \
  .load("dbfs:/databricks-datasets-private/ML/stanford_drone/videos/") \
  .select("path", "modificationTime", "length")

# COMMAND ----------

display(vids)

# COMMAND ----------

silver = bronze.withColumn("img_path", extract_frames(col("path")))
display(silver)

# COMMAND ----------

silver.count()

# COMMAND ----------

df1 = silver
display(df1)

# COMMAND ----------

df1.count()

# COMMAND ----------

df2 = df1.withColumn("path", regexp_replace("img_path", "/dbfs/", "dbfs:/"))
display(df2)

# COMMAND ----------

df2.count()

# COMMAND ----------

df3 = spark.read.format("binaryFile").option("pathGlobFilter", "*{}".format(".jpg")).load(IMG_OUTPUT_DIR).withColumn("content_hash", sha2("content", 512))
display(df3)

# COMMAND ----------

df3.count()

# COMMAND ----------

df4 = df3.join(df2, "path")
display(df4)
