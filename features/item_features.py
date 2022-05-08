# Load libraries ---------------------------------------------

from datetime import datetime, timedelta
from dateutil.easter import easter
from data_preprocessing.dataset_specification import DatasetSpecification
from functools import reduce

import pandas as pd
import numpy as np
# ------------------------------------------------------------

class ItemFeatures(object):
  
    ###########################
    # Item features functions #
    ###########################

  @staticmethod
  def is_cheapest_and_is_most_expensive(items_df):
    """
    Returns a dataframe with the is_cheapest and is_most_expensive features.
    """

    items_df['is_cheapest'] = items_df['room_segment'].apply(lambda x: 1 if x == '[0-160]' else 0)
    items_df['is_most_expensive'] = items_df['room_segment'].apply(lambda x: 1 if x == '[500-900]' else 0)

    return ['is_cheapest', 'is_most_expensive'], items_df

  @staticmethod
  def most_frequent_number_of_people(items_df):
    """
    Returns a dataframe with the most_frequent_number_of_people feature.
    """

    bucket_to_numeric_dict = {
      '[0-1]': 0.5,
      '[1-1]': 1,
      '[2-2]': 2,
      '[2-3]': 2.5,
      '[3-4]': 3.5,
      '[4-7]': 5.5,
      '[5-inf]': 5.5,
      '[8-inf]': 10.5,
      '[0-160]': 80,
      '[160-260]': 210,
      '[260-360]': 310,
      '[360-500]': 430,
      '[500-900]': 700,
    }

    def bucket_to_numeric(x):
      return bucket_to_numeric_dict[x] if x in bucket_to_numeric_dict else x

    items_df['most_frequent_number_of_people'] = items_df.groupby('item_id')['n_people_bucket'].transform(lambda x: x.value_counts().index[0])
    items_df['most_frequent_number_of_people'] = items_df['most_frequent_number_of_people'].apply(bucket_to_numeric)

    return ['most_frequent_number_of_people'], items_df

  @staticmethod
  def is_item_most_popular(items_df):
    """
    Returns a dataframe with the is_most_popular feature.
    """

    most_frequent = items_df['item_id'].mode()[0]

    items_df['is_most_popular'] = items_df['item_id'].apply(lambda x: 1 if x == most_frequent else 0)

    return ['is_most_popular'], items_df

