import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


########################################################
# Implement here test_row_count and test_price_range   #
########################################################

def test_row_count(data: pd.DataFrame, expected_row_count: int):
    """
    Test that the number of rows in the data matches the expected row count.
    """
    actual_row_count = len(data)
    assert actual_row_count == expected_row_count, f"Expected {expected_row_count} rows, but got {actual_row_count}."
    
def test_price_range(data: pd.DataFrame, min_price: float, max_price: float):
    """
    Test that all prices in the data are within the specified range.
    """
    actual_prices = data['price']
    assert actual_prices.between(min_price, max_price).all(), f"Prices are not within the range {min_price} to {max_price}."
    assert actual_prices.min() >= min_price, f"Minimum price {actual_prices.min()} is below the expected minimum {min_price}."
    assert actual_prices.max() <= max_price, f"Maximum price {actual_prices.max()} is above the expected maximum {max_price}."

