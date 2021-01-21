!pip install keras
!pip install tensorflow

 ### Import all library that i will use it



import keras
import sklearn
import math
import random
from matplotlib import pyplot
from operator import itemgetter

### Load ,train+test and show the dataset Fashion-mnist

fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images, test_labels)= fashion_mnist.load_data()

for i in range(9):
    #define subplot
    pyplot.subplot(330 + 1 + i)
    #plot raw pixel data
    pyplot.imshow(train_images[i],cmap=pyplot.get_cmap('gray'))
#show the figure
pyplot.show()

train_images=train_images.reshape(train_images.shape[0],-1)
train_images1=train_images.tolist()
print(len(train_images1))
print(len(train_images1[0]))
train_labels1=train_labels.tolist()
print(len(train_labels1))
k=3

### Creating variables and assigning values



def Count_image(lebal_id,liist):
    return [b for a,b in liist].count(lebal_id) #Comprehension List

### Compute Euclidean

def compute_Euclidean(image1, image2):
    distance = 0
    for a in range(len(image1)-1):
        distance = distance+pow((image1[x] - image2[x]), 2)
    return math.sqrt(distance)


### Apply ENN 

    
 

def find_ENN(img,lst, k):
    distances = []
    for x in range(len(lst)):
        dist = compute_Euclidean(img, lst[x][0])
        distances.append((lst[x], dist))

   

    distances.sort(key= itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])  
    return neighbors

### Find The Majority

def Find_majority_of_list(lstt): 
    lstlbl=Extract(lstt)
    return max(set(lstlbl), key = lstlbl.count) 

def Extract(lstt): 
    return [item[1] for item in lstt] #Comprehension List

def most_frequent_test(_lst): 
    lstlbl=[item[1] for item in _lst] #Comprehension List
    return max(set(lstlbl), key = lstlbl.count) 

### Apply Zip then shuffle the list

zipped=zip(train_images1,train_labels1)
zip_list=list(zipped)
random.shuffle(zip_list)
print(len(zip_list))

### We have 3000 image and each class have 300 image 300*10 class=3000

labelss=[]
imagess=[]
count=0
for x in range(10):
    count=0
    for img,lbl in zip_list:
        if(lbl==x):
            labelss.append(lbl)
            imagess.append(img)
            count+=1
        if(count==300):
                break
final_result=list(zip(imagess,labelss))
print(len(final_result))



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for y in range(10):
    print("Class Name " + str(y) +" : "+ class_names[y] +" = "+str(labelss.count(y)))

### The Majority without repated 
Because the dataset is too large i filter it into small to validation of code

all_majority=[] 
Majority=set() 
images_=10     
all_images=500  

random.shuffle(final_result)

for img,lbl in final_result[:images_]:
    all_majority=find_ENN(img,zip_list[:all_images],k)
    Majority.add(Find_majority_of_list(all_majority))

print(Majority)


### To check of 3000 if the label of image in the Majority add it to the list else remove it

FinalListMajority=[]

for item in final_result:

        # Lambda Expression
    check=lambda x : x in Majority     
    if(check(item[1])):
        FinalListMajority.append(item)
        
print(len(FinalListMajority))        
print(len(final_result))

# Final Result

imgs,lbls=zip(*FinalListMajority)

for x in range(10):
    print("Class Name " + str(x) +": "+ class_names[x] +"  = "+str(lbls.count(x)))
    
