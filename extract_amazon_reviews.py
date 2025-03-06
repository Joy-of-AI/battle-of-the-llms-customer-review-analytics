"""
Extract customer reviews for Amazon Shopping Apps from Google Play and App Store.

This script retrieves and processes app reviews from both platforms, formatting the data into a structured format for analysis.

Author: Amir Amin
Version: 1.0
Last Updated: 2025-03-08
"""

from google_play_scraper import Sort, reviews
from app_store_scraper import AppStore
import pandas as pd
from tabulate import tabulate
import numpy as np
import json, os, uuid
from datetime import datetime, timedelta

# ---------------------------------------------------------------------
# Section 1: Extract Amazon Shopping reviews from Google Play
# ---------------------------------------------------------------------

# Define the Google Play app ID for Amazon Shopping
app_id = "com.amazon.mShop.android.shopping"

# Fetch reviews sorted by newest (limit is defined by max_reviews)
g_reviews, _ = reviews(
    app_id,
    lang='en',
    country='us',
    sort=Sort.NEWEST
    # max_reviews can be specified if a limit is desired
)

# ---------------------------------------------------------------------
# Section 2: Extract Amazon Shopping reviews from Apple App Store
# ---------------------------------------------------------------------

# Define the app ID for Amazon Shopping on the Apple App Store
a_reviews = AppStore(country='us', app_name='amazon-shopping', app_id='297606951')
a_reviews.review()  # Fetch reviews

# ---------------------------------------------------------------------
# Section 3: Check for available reviews and print column names for both of Google Play and Apple Store
# ---------------------------------------------------------------------

# Check if Google Play reviews exist before printing column names
print("-" * 150)  # Prints a dashed line of 50 dashes

if g_reviews:
    print("Google Play Review Columns:", g_reviews[0].keys())
else:
    print("No Google Play reviews available.")

# Check if Apple App Store reviews exist before printing column names
if a_reviews.reviews:
    print("Apple App Store Review Columns:", a_reviews.reviews[0].keys())
else:
    print("No Apple App Store reviews available.")

# ---------------------------------------------------------------------
# Section 4: Transform data structure for uniformity and union
# ---------------------------------------------------------------------

# Prepare Google Play reviews for data transformation
g_reviews_data = pd.DataFrame(g_reviews)  # Convert Google Play reviews into DataFrame
g_reviews_data['date'] = pd.to_datetime(g_reviews_data['at']).dt.date  # Extract only the date
g_reviews_data['review'] = g_reviews_data['content']  # Rename 'content' to 'review'
g_reviews_data['title'] = g_reviews_data['reviewCreatedVersion'].fillna('No title')  # Add title (default 'No title')
g_reviews_data['userName'] = g_reviews_data['userName']  # Keep the userName
g_reviews_data['platform'] = 'Google Play'  # Flag as Google Play reviews
g_reviews_data['is_replied'] = g_reviews_data['replyContent'].notna().replace({True: 'Yes', False: 'No'})  # Check for reply

# Filter relevant columns for Google Play reviews
g_reviews_data = g_reviews_data[['date', 'userName', 'platform', 'review', 'is_replied']]

# Prepare App Store reviews for data transformation
a_reviews_data = pd.DataFrame(a_reviews.reviews)  # Convert App Store reviews into DataFrame
a_reviews_data['app'] = 'iOS'  # Flag as iOS reviews
a_reviews_data['platform'] = 'App Store'  # Flag as App Store reviews
a_reviews_data['date'] = pd.to_datetime(a_reviews_data['date']).dt.date  # Extract only the date
a_reviews_data['review'] = a_reviews_data['review']  # Rename 'review' to 'review'
a_reviews_data['title'] = a_reviews_data['title'].fillna('No title')  # Add title (default 'No title')
a_reviews_data['userName'] = a_reviews_data['userName']  # Keep the userName
a_reviews_data['is_replied'] = 'No'  # App Store reviews data doesn't include any olumns showing replies. So, assuming no replies for these reviews.

# Filter relevant columns for App Store reviews
a_reviews_data = a_reviews_data[['date', 'userName', 'platform', 'review', 'is_replied']]

# Combine both Google Play and Apple Store reviews into one dataset
combined_reviews = pd.concat([g_reviews_data, a_reviews_data], ignore_index=True)

# Test combined data- Display 5 sample reviews from each platforms (Apple and Google)
ios_reviews_sample = combined_reviews[combined_reviews['platform'] == 'App Store'].head(5)
google_reviews_sample = combined_reviews[combined_reviews['platform'] == 'Google Play'].head(5)

# App Store includes 8M+ revies and Google Play has 4M+ reviews
# For purpose of this project, we consider 5 sample data from each of these platforms
# Going forward, all analytics will be implemented to this sample dataset 
final_sample = pd.concat([ios_reviews_sample, google_reviews_sample], ignore_index=True)

# Display final results in a tabular format
print("-" * 150)  # Prints a dashed line of 50 dashes
print(tabulate(final_sample, headers="keys", tablefmt="grid"))
