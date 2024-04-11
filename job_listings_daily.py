import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import re

def fetch_job_listings(geoareaid, page_limit=1):
    base_url = 'https://www.jobindex.dk/jobsoegning.json'
    job_listings = []

    # Calculate yesterday's date
    yesterday_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    for page in range(1, page_limit+1):
        params = {
            'geoareaid': geoareaid,
            'page': page
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            html_content = data.get('result_list_box_html', '')
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find all job ads
            job_ads = soup.find_all('div', class_='jobsearch-result')
            for job_ad in job_ads:
                title = job_ad.find('div', class_='jobad-element-menu-share')['data-share-title']
                url = job_ad.find('div', class_='jobad-element-menu-share')['data-share-url']
                entry_id = url.replace("https://www.jobindex.dk/vis-job/", "")  # Extracting ID
                
                # Extracting location
                location_tag = job_ad.find('div', class_='jobad-element-area')
                location = location_tag.find('span', class_='jix_robotjob--area').text.strip() if location_tag and location_tag.find('span', class_='jix_robotjob--area') else 'unknown'
                
                # Extracting published date
                published_tag = job_ad.find('time')
                published_date = published_tag['datetime'] if published_tag else 'unknown'
                
                # Extracting description
                description_tag = job_ad.find('div', class_='PaidJob-inner')
                description = description_tag.find('p').get_text() if description_tag else 'No description available'

                # Compare published date with yesterday's date
                if published_date.startswith(yesterday_date):
                    job_listings.append([entry_id, title, description, published_date, location, url]) 
                else:
                    # Since the posts are ordered by date, if we hit a post that is not from yesterday, we can stop.
                    break
        else:
            break
    
    return job_listings

# Function to remove special characters from a string
def remove_special_characters(text):
    # Define a regular expression pattern to match non-alphanumeric characters
    pattern = r'[^a-zA-Z0-9\s]'  # Matches any character that is not a letter, digit, or whitespace

    # Use the sub() function from the re module to replace matched characters with an empty string
    clean_text = re.sub(pattern, '', str(text))  # Convert NaN to string before applying the function
    return clean_text

# Gettings yesterdays job listings
geoareaid = ''
job_listings = fetch_job_listings(geoareaid, page_limit=100)
df = pd.DataFrame(job_listings, columns=['Entry_id', 'Title', 'Description', 'Published Date', 'Location', 'URL'])

<<<<<<< HEAD
# Remove special characters from the "Title" column
df['Title'] = df['Title'].apply(remove_special_characters)
=======
# saving to csv
csv_filename = 'job_listings_full.csv'
df.to_csv(csv_filename, index=False)

# URL to upload the CSV file
github_raw_url = 'https://raw.githubusercontent.com/jogfx/MLops-exam/main/'

# Upload the CSV file to GitHub
with open(csv_filename, 'rb') as file:
    response = requests.put(github_raw_url + csv_filename, data=file)

# Check if upload was successful
if response.status_code == 200:
    print("CSV file uploaded successfully to GitHub.")
else:
    print("Failed to upload CSV file to GitHub. Status code:", response.status_code)
>>>>>>> cb0e1b4444f19a1f317a5375204380b2987254c3

# Remove special characters from the "Description" column, handling NaN values
df['Description'] = df['Description'].fillna('').apply(remove_special_characters)

# saving to csv
csv_filename = 'job_listings_cleaned.csv'
df.to_csv(csv_filename, index=False)

# URL to upload the CSV file
github_raw_url = 'https://raw.githubusercontent.com/jogfx/MLops-exam/main/'

# Upload the CSV file to GitHub
with open(csv_filename, 'rb') as file:
    response = requests.put(github_raw_url + csv_filename, data=file)

# Check if upload was successful
if response.status_code == 200:
    print("CSV file uploaded successfully to GitHub.")
else:
    print("Failed to upload CSV file to GitHub. Status code:", response.status_code)

df.head()