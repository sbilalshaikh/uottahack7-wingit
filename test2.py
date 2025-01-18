import os
import requests
from bs4 import BeautifulSoup

GOOGLE_IMAGE = 'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&q='

# The User-Agent request header contains a characteristic string 
# that allows the network protocol peers to identify the application type,
# operating system, and software version of the requesting software user agent.
# This is needed for google search
usr_agent = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
}

SAVE_FOLDER = 'OtherImages'
filename_tosave = "image"
dir = ""


def main():
    if not os.path.exists(SAVE_FOLDER):
        print("Making Directory")
        os.mkdir(SAVE_FOLDER)
    print("Directory exists")
    download_images()


def download_images():
    data = input('What are you looking for? ')

    print('Start searching...')
    
    searchurl = GOOGLE_IMAGE + data
    print(f"Searching URL: {searchurl}")

    # Request URL, without User-Agent the permission gets denied
    response = requests.get(searchurl, headers=usr_agent)
    html = response.text
    
    # Parsing HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Retrieve image links
    image_links = []
    
    # Look for 'img' tags and extract the 'src' attribute
    for img_tag in soup.find_all('img'):
        img_url = img_tag.get('src')
        if img_url and img_url.startswith('http'):
            image_links.append(img_url)

    # Download images
    i = 0
    for img_url in image_links[:5]:  # Limiting to the first 5 images
        i += 1
        try:
            response = requests.get(img_url)
            imagename = os.path.join(SAVE_FOLDER, f'{filename_tosave}_{i}.jpg')
            with open(imagename, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded image {i}: {img_url}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image {img_url}: {e}")

    print('Done downloading images.')


if __name__ == '__main__':
    main()
