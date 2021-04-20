import os
import json
import requests
from bs4 import BeautifulSoup
import shutil
import urllib

#must specify page numbers
path = os.path.dirname(os.path.realpath(__file__))
directory = os.getcwd()
page_numbers = 80
page_num = 0

file_count = 0

model_folder = path + "\model_folder"
data_folder = path + "\data_folder"
image_folder = path + "\image_folder"

if os.path.isdir(model_folder):
    pass
else:
    os.mkdir(model_folder)
if os.path.isdir(data_folder):
    pass
else:
    os.mkdir(data_folder)
if os.path.isdir(image_folder):
    pass
else:
    os.mkdir(image_folder)

for i in range(page_numbers):
    page_num += 1
    if i == 0:
        archive = 'https://www.bikecad.ca/archive'
    else:
        archive = 'https://www.bikecad.ca/archive?page={}'.format(i)

    archivepage = requests.get(archive)

    archive_soup = BeautifulSoup(archivepage.content, 'html.parser')

    results = archive_soup.find(id = 'content')

    for link in results.find_all('a'):
        link = link.get('href')
        #checks that link is to a bike model
        if str(link)[0:6] != 'https:':
            pass
        else:
            URL = str(link)
            name = URL.rpartition('/')[-1]

            #checks if file already exists in directory
            if os.path.isfile(model_folder + '\{}.bcad'.format(name)):
                #print('File already downloaded.')
                pass
            else:

                #if not, proceeds to parse and find download link for model
                r = requests.get(URL)   
                r_soup = BeautifulSoup(r.content, 'html.parser')
                content = r_soup.find(id = 'content')
                
                #downloads image file
                image = content.find_all('img')
                for line in image:
                    img_link = line.get('src')
                    if name in img_link:
                        pic = requests.get(img_link)
                        os.chdir(image_folder)
                        with open("{}_image.png".format(name), 'wb') as im:
                            im.write(pic.content)
                        os.chdir(path)


                #BCAD file download
                for download in content.find_all('a'):
                    download = download.get('href')

                    #ensures that the link is specifically for the download
                    if str(download)[-5:] != '.bcad':
                        continue
                    else:

                        #actually downloads
                        download_link = requests.get(download)
                        os.chdir(model_folder)
                        with open("{}.bcad".format(name), 'wb') as f:
                            f.write(download_link.content)
                        os.chdir(path)

                        #collects data from the website
                        title = r_soup.find('h2', 'node__title node-title')
                        author = r_soup.find('a', 'username')
                        bike_info = {}
                        bike_info['Title'] = title.string
                        bike_info['Link'] = URL
                        bike_info['Author Username'] = author.string


                        #'check' checks for a link to a previous model; different format so must
                        # be accounted for
                        check = 0

                        data = r_soup.find(id  = "bcminfo")

                        #builds dictionary with data
                        for i in range(len(list(data.find_all('div', 'field-label')))):
                            label = data.find_all('div', 'field-label')[i]
                            label = label.string.rpartition(':')[0]
                            if label == "Started from":
                                if len(data.find_all('a')) >= 1:
                                    item = str(data.find_all('a')[0].get('href'))
                                    check = 1
                                else:
                                    item = 'website not provided'
                                    check = 1
                            else:
                                if check == 1:
                                     i -= 1
                                item = data.find_all('div', 'field-item even')[i]
                                item = str(item.string)
                            bike_info[label] = item

                        #writes to json file
                        os.chdir(data_folder)
                        with open('{}_data.txt'.format(name), 'w') as d:
                            json.dump(bike_info, d)
                        os.chdir(path)
                        file_count += 1
                        print('Bike ' + str(file_count) + " successfully downloaded.")
                        
    print("Page " + str(page_num) + " successfully downloaded! "
         + str(page_numbers - page_num) + " pages left." )                    
























