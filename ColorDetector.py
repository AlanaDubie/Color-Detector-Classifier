# -*- coding: utf-8 -*-
"""Color_Detect.ipynb

"""
import pandas as pd
import cv2

#color dominance imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


test_img = cv2.imread("color_image.jpg")

index=["color_name", "hex", "R", "G", "B", "color_class"]
csv = pd.read_csv('colors.csv',names=index, header=None, skiprows=[0])
#csv.drop(csv.tail(5000).index, inplace = True)

clicked = False
r = g = b = xpos = ypos = 0

def recognize_color(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        #finds the nearest color using RGB Values (k nearest neighbor algorithm)
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            color_name = csv.loc[i,"color_name"]
    return color_name

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global b,g,r,xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = test_img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)

def dominant_colors(test_img):
    #need to convert it from BGR to RGB
    test_img =cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
    #plt.imshow(test_img)
    #reshape image to array to shape (H*W, 3)
    test_img=test_img.reshape((test_img.shape[1]*test_img.shape[0],3))
    
    #sets number of clusters (number of dominate colors) to 6
    kmeans=KMeans(n_clusters=6)
    s=kmeans.fit(test_img)
    labels=kmeans.labels_
   # print(labels)
    labels=list(labels)
    
    #determines number of centroids of clusters for RGB pixel
    centroid=kmeans.cluster_centers_
    #centroid holds the rgb values of each cluster

    #calculates the percentage of each cluster and
    percent=[]
    dom_colors = []
    for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)
        
    #scans ndarray of centroid, and gets rgb values of each centroid
    for x in (centroid):
        red = int(x[0])
        green = int(x[1])
        blue = int(x[2])
        dom_colors.append(recognize_color(red,green,blue))    
    #print(tab)
    

   #plt donut chart
    plt.pie(percent, colors=np.array(centroid/255), startangle=90)
    #draw inner white circle
    centre_circle = plt.Circle((0,0),0.50,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    #plt legend & title
    plt.figtext(0.3, 0.45, 'Dominant \n Colors', fontsize=16, fontweight='light', color='black')
    #rounds to percent and concatenates to a string
    round_percent = ["{:.0%}".format(element) for element in percent]
    #formats legend labels: dominate color, percent
    legend_labels = [f'{l}, {s}' for l, s in zip(dom_colors, round_percent)]
    plt.legend(legend_labels,loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    
    #print chart
    plt.tight_layout()
    return plt.savefig('ColorWheel.jpg',dpi=600, bbox_inches='tight')
    

dominant_colors(test_img)
cv2.namedWindow('Color Dectection')
cv2.setMouseCallback('Color Dectection', mouse_click)


while True:
    cv2.imshow('Color Dectection',test_img)
    if clicked:
        cv2.rectangle(test_img,(20,20), (750,60), (b,g,r), -1)
        #Creates text string to display the color names and RGB values 
        text = recognize_color(r,g,b) + ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b) 
        
        cv2.putText(test_img, text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)
        #displays text in black for light colors
        if(r+g+b>=600):
            cv2.putText(test_img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
            
        clicked=False
        
    #Break the loop when 'esc' key is pressed
    if cv2.waitKey(20) & 0xFF ==27:
        break
    
cv2.destroyAllWindows()
cv2.waitKey(1)
