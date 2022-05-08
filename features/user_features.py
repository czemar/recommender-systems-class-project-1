# Load libraries ---------------------------------------------

from datetime import datetime, timedelta
from dateutil.easter import easter
from data_preprocessing.dataset_specification import DatasetSpecification
from functools import reduce

import pandas as pd
import numpy as np
# ------------------------------------------------------------

class UserFeatures(object):

  ###########################
  # User features functions #
  ###########################

  @staticmethod
  def most_popular_user_features_one_hot_encoding(users_df):
    """
    Returns a dataframe with the most popular user features one-hot encoded.
    """

    # Get the most popular term
    most_popular_term = users_df['term'].value_counts().index[0]
    most_popular_length_of_stay_bucket = users_df['length_of_stay_bucket'].value_counts().index[0]
    most_popular_rate_plan = users_df['rate_plan'].value_counts().index[0]
    most_popular_room_segment = users_df['room_segment'].value_counts().index[0]
    most_popular_n_people_bucket = users_df['n_people_bucket'].value_counts().index[0]
    most_popular_weekend_stay = users_df['weekend_stay'].value_counts().index[0]

    users_df['most_popular_term'] = pd.to_numeric(users_df['term'].apply(lambda x: 1 if x == most_popular_term else 0))
    users_df['most_popular_length_of_stay_bucket'] = pd.to_numeric(users_df['length_of_stay_bucket'].apply(lambda x: 1 if x == most_popular_length_of_stay_bucket else 0))
    users_df['most_popular_rate_plan'] = pd.to_numeric(users_df['rate_plan'].apply(lambda x: 1 if x == most_popular_rate_plan else 0))
    users_df['most_popular_room_segment'] = pd.to_numeric(users_df['room_segment'].apply(lambda x: 1 if x == most_popular_room_segment else 0))
    users_df['most_popular_n_people_bucket'] = pd.to_numeric(users_df['n_people_bucket'].apply(lambda x: 1 if x == most_popular_n_people_bucket else 0))
    users_df['most_popular_weekend_stay'] = pd.to_numeric(users_df['weekend_stay'].apply(lambda x: 1 if x == most_popular_weekend_stay else 0))

    return ['most_popular_term', 'most_popular_length_of_stay_bucket', 'most_popular_rate_plan', 'most_popular_room_segment', 'most_popular_n_people_bucket', 'most_popular_weekend_stay'], users_df

  @staticmethod
  def features_propability_distribution(users_df):
    """
    Returns a dataframe with the features propability distribution.
    """
      
    # Get the features propability distribution
    users_df['term_propability_distribution'] = pd.to_numeric(users_df['term'].apply(lambda x: users_df['term'].value_counts()[x] / users_df['term'].value_counts().sum()))
    users_df['length_of_stay_bucket_propability_distribution'] = pd.to_numeric(users_df['length_of_stay_bucket'].apply(lambda x: users_df['length_of_stay_bucket'].value_counts()[x] / users_df['length_of_stay_bucket'].value_counts().sum()))
    users_df['rate_plan_propability_distribution'] = pd.to_numeric(users_df['rate_plan'].apply(lambda x: users_df['rate_plan'].value_counts()[x] / users_df['rate_plan'].value_counts().sum()))
    users_df['room_segment_propability_distribution'] = pd.to_numeric(users_df['room_segment'].apply(lambda x: users_df['room_segment'].value_counts()[x] / users_df['room_segment'].value_counts().sum()))
    users_df['n_people_bucket_propability_distribution'] = pd.to_numeric(users_df['n_people_bucket'].apply(lambda x: users_df['n_people_bucket'].value_counts()[x] / users_df['n_people_bucket'].value_counts().sum()))
    users_df['weekend_stay_propability_distribution'] = pd.to_numeric(users_df['weekend_stay'].apply(lambda x: users_df['weekend_stay'].value_counts()[x] / users_df['weekend_stay'].value_counts().sum()))

    return ['term_propability_distribution', 'length_of_stay_bucket_propability_distribution', 'rate_plan_propability_distribution', 'room_segment_propability_distribution', 'n_people_bucket_propability_distribution', 'weekend_stay_propability_distribution'], users_df

  @staticmethod
  def features_averange(users_df):
    """
    Returns a dataframe with the features averange from all reservations of the user.
    """

    users_df_copy = users_df.copy()

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

    users_df_copy['length_of_stay_bucket'] = users_df_copy['length_of_stay_bucket'].apply(bucket_to_numeric)
    users_df_copy['n_people_bucket'] = users_df_copy['n_people_bucket'].apply(bucket_to_numeric)
    users_df_copy['room_segment'] = users_df_copy['room_segment'].apply(bucket_to_numeric)

    users_df_copy['length_of_stay_bucket'] = pd.to_numeric(users_df_copy['length_of_stay_bucket'], errors='coerce')
    users_df_copy['n_people_bucket'] = pd.to_numeric(users_df_copy['n_people_bucket'], errors='coerce')
    users_df_copy['room_segment'] = pd.to_numeric(users_df_copy['room_segment'], errors='coerce')

    users_df['length_of_stay_averange'] = users_df_copy.groupby('user_id')['length_of_stay_bucket'].transform(lambda x: x.mean())
    users_df['room_segment_averange'] = users_df_copy.groupby('user_id')['room_segment'].transform(lambda x: x.mean())
    users_df['n_people_averange'] = users_df_copy.groupby('user_id')['n_people_bucket'].transform(lambda x: x.mean())

    return ['length_of_stay_averange', 'room_segment_averange', 'n_people_averange'], users_df

  @staticmethod
  def most_popular_reservation_similarity(users_df):
    """
    Returns a dataframe with the most popular reservation similarity.
    """

    # Get the most popular term
    most_popular_term = users_df['term'].value_counts().index[0]
    most_popular_length_of_stay_bucket = users_df['length_of_stay_bucket'].value_counts().index[0]
    most_popular_rate_plan = users_df['rate_plan'].value_counts().index[0]
    most_popular_room_segment = users_df['room_segment'].value_counts().index[0]
    most_popular_n_people_bucket = users_df['n_people_bucket'].value_counts().index[0]
    most_popular_weekend_stay = users_df['weekend_stay'].value_counts().index[0]

    weights = [0.2, 0.2, 0.1, 0.2, 0.15, 0.15]

    # Compare most popular combination with each reservation
    users_df['most_popular_similarity'] = (
      users_df['term'] == most_popular_term
    ) * weights[0] + (
      users_df['length_of_stay_bucket'] == most_popular_length_of_stay_bucket
    ) * weights[1] + (
      users_df['rate_plan'] == most_popular_rate_plan
    ) * weights[2] + (
      users_df['room_segment'] == most_popular_room_segment
    ) * weights[3] + (
      users_df['n_people_bucket'] == most_popular_n_people_bucket
    ) * weights[4] + (
      users_df['weekend_stay'] == most_popular_weekend_stay
    ) * weights[5]

    return ['most_popular_similarity'], users_df
