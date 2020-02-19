#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:24:36 2020

to install selenium run:
pip install selenium

to get chromium driver run:
sudo apt-get install chromium-chromedriver
check 'which chromedrive' in terminal for chromepath

Need to add the loops to build and append the pickle DF

@author: mogmelon
"""

from selenium import webdriver
import json
import pandas as pd, numpy as np
import os
import requests
import datetime
from PIL import Image
import time

def scrape(search, folder_name):
    #Set run variables
    chromepath='/usr/bin/chromedriver'
    search=search
    savedir=folder_name
    melons=['watermelon', 'honeydew', 'canteloupe']
    max_images=2500
    url="http://www.bing.com/images/search?q=" + search + "&FORM=HDRSC2"
    #now=datetime.datetime.now()
    #now_str=now.strftime("%Y%m%dT%H%M%S")
    datadir=('/home/mogmelon/Python/Data/melonID/%s/' % (savedir))
    
    #Initiate Selenium browser    
    browser = webdriver.Chrome(chromepath)
    browser.get(url)
    
    #Create a DataFrame to store links and image data. 
    img = pd.DataFrame()
    
    #for _ in range(1000):
    #    browser.execute_script("window.scrollBy(0,1000000)")
    
    #Infinite Scrape
    number_of_scrolls=100
    for _ in range(int(number_of_scrolls)):
        for __ in range(10):
            # multiple scrolls needed to show all 400 images
            browser.execute_script("window.scrollBy(0, 1000000)")
            time.sleep(0.2)
        # to load next 400 images
        time.sleep(0.5)
        try:
                browser.find_element_by_xpath("//input[@value='See more images']").click()
        except Exception as e:
                print ("Less images found: {}".format(e))
                break
    
    #Find the image content and populate the DataFrame
    test=browser.find_elements_by_xpath('//a[contains(@class,"iusc")]')
    for x in test:
        try:
            img_link    =(json.loads(x.get_attribute('m'))["murl"])
            site        =(json.loads(x.get_attribute('m'))["purl"])
            img_desc    =(json.loads(x.get_attribute('m'))["desc"])
    
            if img_link.split(".")[-1] in ('jpg','jpeg'):
                imgtype = 'jpg'   
            elif img_link.split(".")[-1] in ('png'):
                imgtype = 'png'   
            else:
                continue
            
            temp    ={'image': [img_link], 'image_type': [imgtype], 'image_desc': [img_desc], 'site': [site], 'label': [search]}
            temp_df =pd.DataFrame(data=temp)
            img     =img.append(temp_df)
            print('Added link: ', img_link)
    
        except:
            print('no data found in:', x)
            np.nan
        
        if (max_images==None):
            pass  
        elif len(img.index) > max_images:
                    break
        
    #Prepare to save the images     
    img.index = range(len(img.index))
    counter = 1
    succounter = 0
    img['image_ref']=None
    
    #Make the storage directory
    if not os.path.exists(datadir):
            os.makedirs(datadir)
    
    #Save the images
    for i in range(len(img)):
        #Check the images are .jpg or .png for use in Keras
        if img['image_type'][i] in ('jpg','png'):
            #Create the filename and add a ref to DF
            filename=str(datadir + img['label'][i] + '_' + str(img.index[i]) + '.' + img['image_type'][i])
            img['image_ref'][i]=filename
            
            try:
                #Get the image
                img['image_ref'][i]=filename
                r = requests.get(img['image'][i], timeout=5)
                print('Saved image ', filename)
                with open(filename, 'wb') as f:
                    f.write(r.content)
      
                try:
                    #Check the image is an image file, remove if not able to open 
                    im=Image.open(filename)
                    succounter += 1
                    #image_number +=1
                    print('Successfully retrieved %s/%s images of %s possible links' % (succounter, counter, len(img.index)))
                except IOError:
                    print(filename + ' not an image file')
                    os.remove(filename)
                   
            except:
                #Tells us if the image is irretrievable - should drop the row in future
                img['image_ref'][i]='Fail'
                print("Image not retrievable: " + img['image'][i])  
        else:
            #Catches non .jpg/.png - This is fixed in the initial link extraction now UPDATE.
            img['image_ref'][i]='Fail'
            print("Cannot use image type in Keras: " + img['image'][i]) 
        #Increment the trials counter    
        counter += 1      
        
    print(succounter, "pictures succesfully downloaded")
    browser.close() 
    
    #Save the DataFrame for incremental downloads
    #img.to_pickle(datadir+search+'.pickle')

"""   
    search_terms= {'watermelon_':['watermelon', 'red+melon', 'watermelon+melon+cartoon', 'watermelon+melon+clipart', 'watermelon+melon+art', \
                                  'watermelon+melon+drawing', 'watermelon+melon+chunks', 'watermelon+melon+smoothie', 'watermelon+melon+dessert', 'watermelon+melon+field', \
                                  'watermelon+melon+salad','watermelon+melon+shop', 'red+melon+art',  'red+melon+drawing', 'red+melon+clipart', \
                                  'red+melon+salad'],
                   'canteloupe_':['canteloupe', 'orange+melon', 'canteloupe+melon+cartoon', 'canteloupe+melon+clipart', 'canteloupe+melon+art', \
                                  'canteloupe+melon+drawing', 'canteloupe+melon+chunks', 'canteloupe+melon+smoothie', 'canteloupe+melon+dessert', 'canteloupe+melon+field', \
                                  'canteloupe+melon+salad','canteloupe+melon+shop', 'orange+melon+art',  'orange+melon+drawing', 'orange+melon+clipart', \
                                  'orange+melon+salad']\
                       
                   'honeydew_': ['honeydew', 'yellow+melon', 'honeydew+melon+cartoon', 'honeydew+melon+clipart', 'honeydew+melon+art', \
                                 'honeydew+melon+drawing', 'honeydew+melon+chunks', 'honeydew+melon+smoothie', 'honeydew+melon+dessert', 'honeydew+melon+field', \
                                 'honeydew+melon+salad','honeydew+melon+shop', 'yellow+melon+art',  'yellow+melon+drawing', 'yellow+melon+clipart', \
                                 'yellow+melon+salad']\
                   'random_':    ['people', 'food', 'landscape', 'animals', 'mountains', 'buildings', 'plants', 'art', 'science', 'history', \
                                  'cars', 'electronics', 'fruit', 'clothes', 'interior+design', ]\
                       }
"""

def single_search():
    search_terms=['honeydew', 'yellow+melon', 'honeydew+melon+cartoon', 'honeydew+melon+clipart', 'honeydew+melon+art', \
                  'honeydew+melon+drawing', 'honeydew+melon+chunks', 'honeydew+melon+smoothie', 'honeydew+melon+dessert', 'honeydew+melon+field', \
                  'honeydew+melon+salad','honeydew+melon+shop', 'yellow+melon+art',  'yellow+melon+drawing', 'yellow+melon+clipart', \
                  'yellow+melon+salad']
    folder_name='honeydew_'
    
    for i,j in enumerate(search_terms):
        print('Creating folder %s out of %s' % (i,len(search_terms)))
        scrape(j, folder_name+str(i))

def multi_search():
    
    search_terms= {'watermelon_':['woman+with+watermelon', 'man+with+watermelon', 'eating+watermelon', 'woman+holding+watermelon'],\
                   'canteloupe_':['woman+with+canteloupe', 'man+with+canteloupe', 'eating+canteloupe+melon', 'woman+holding+canteloupe+melon'],\
                   'honeydew_':['woman+with+honeydew', 'man+with+honeydew','eating+honeydew+melon', 'woman+holding+honeydew+melon'],\
                   'random_':['lake', 'rivers', 'vegetables', 'clothes']\
                   }
                   
    for search, terms in search_terms.items():
        print(search) 
        for i, query in enumerate(terms):
            print('Creating folder %s out of %s' % (i,len(terms)))
            scrape(query, search+'1_'+str(i))

