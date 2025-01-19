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

def imageGen(search):
    searchurl = GOOGLE_IMAGE + search + " clipart"
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

    return image_links[0]
