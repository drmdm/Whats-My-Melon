# What's My Melon?
## My Introduction to Convolutional Neural Networks - Melon Image Recognition

Following the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) course covering Keras and seeing what CNNs could do I was keen to build a CNN project of my own. What I thought would take a few hours turned into a much deeper project as i quickly moved from a basic CNN to state of the art image-segmentation models. This project documents my progress so far. 

This is how the project unfolded:
1. Web Scraper
2. First Attempt - Basic CNN
3. Second Attempt - Transfer Learning
4. Third Attempt - Mask-RCNN
5. Fourth Attmept- ???

## 1. Web Scraper
I was keen to avoid pre-cleaned, nicely prepared datasets (Built-ins, Kaggle, etc.) as this isn't representative of real projects. It is useful to be able to generate your own datasets and I was keen to build a webscraper. For the basic CNN I was aiming to train on around 2000 images per class with 500 test images for evaluation.  
The initial scraper was built to extract links from Google image search and download the images from the sites. There is no API for extracting images so I built a scraper using Selenium. The initial scraper worked well until I updated Chrome and the HTML metadata had changed breaking the scraper. After a bit of searching I found that Google isn't the easiest site to scrape as they update quite regularly. Bing's HTML seemed to be more static and easier to read so I converted my scraper to use Bing instead. 

## 2. Basic CNN
The basic CNN was based on the lab from the Keras course in the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer). Building the model and training it was straight-forward using 
the Keras ImageDataGenerator. I made a seperate script to evaluate the model and make new predictions. 

## 3. Transfer Learning
Transfer learning uses a pre-trained CNN and allows you to only train the final layer

## 4. Mask-RCNN
After testing the trained CNN on some obvious pictures of melons it did a good job of predicting the various types. The next step was to ask people to send pictures of melons for me to classify. I had overlooked the thought process of the general population and 80% of the images I received were similar to the one below:

This is when I realised the CNN wasn't doing a good job of predicting melons within an image. After a bit of research I realised I need to create an object detection or image segmentation model. One of the most up-to-date methods for this is Mask R-CNN.
I began setting up my model using the package [Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN) and the guides provided in the repository. To create the image masks I chose 120 images of watermelons and created the masks using [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via.html)

## 5. ???
