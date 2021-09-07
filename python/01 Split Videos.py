# Databricks notebook source
# MAGIC %pip install moviepy

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

spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

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
