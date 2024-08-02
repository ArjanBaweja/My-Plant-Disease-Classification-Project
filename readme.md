# Plant Disease Classification Project

This project uses the Plant village detaset and the nvidia jetson nano to run a img classification program that classifys photos of 38 classes of plants gives a output with the class name (the disease the plant or if its healthy) and provides a wiki link you can use to learn more and help your plant.

Input img:
![0c620ec5-11cf-4120-94ab-1311e99df147___FREC_Scab 3131](https://github.com/user-attachments/assets/2546e04e-9033-4075-9ba5-be72e0d5d71d)

(Apple Leaf with Scab Diease)

Output:
![WhatsApp Image 2024-08-03 at 00 20 20_257ecb6a](https://github.com/user-attachments/assets/b83220a7-6c2b-4875-83ed-7a468df7eaa7)

(Correctly Identifyed as Apple__Scab)
(Wiki Link: https://en.wikipedia.org/wiki/Apple_scab)

This project can be used by farmers or just passtionate gardeners which solves the problem of not knowing what to do when you see spots on your plants, with this project you can learn what disease it is or if its just healthy and what further actions you can take to help your plant. The idea of giving a wikipidea is one unique to my project while there are other projects that can identify what plant disease it is not many provide a solution or a next step.

## The Algorithm

The algorithm bhiend this project works in 3 steps. 

The first of which is downloading and formatting the dataset. For this project i used the Plant Village dataset, This dataset has 38 classes of diesases on various plants each class has around 500-1000 imgs. However the orginal PLant vilage dataset is not formated in the correct way for the next steps, beacause to train the ai the dataset needs to be split into Train, Test, Val and a labels.txt file which consits of the names of all 38 classes. The format needs to be split into these 3 classes beacause the ai needs to have a split of photos to trian, use for accuracy and to test. My data was split into a 80,10,10 split with 80% of the data being in the test folder. (this was done using split_datasets.py)

The second step to this algorithim is trainning the model. To train the model you have to get the ai to keep looking at the test imgs to try and learn by itself what type of image belongs in which class the project uses resnet18 for this. then the model uses the val photos to test its accuracy. one cycle of this is called a epoch. for my model i ran 10 epoch however you can run as many as you want i reccomend running more than 25. (this was done using train.py)

Lastly you can run the algrothim with any photo of your choice. The alogorthim will try to fit the img you sent it into one of the classes. and create a pridction. It will then print the prediction along with the name of the class the % sure it is and a link to the wikipidea. This works beacsue the algorithm assignes every class a number from 0-37 based on its position in the labels.txt file in the dataset then whatever it pridects it find the number for that class and saves it. I could then use this number to print the name of the class and a link to the wikipidea by creating two dictonarys one with all links to the winkipidea labeled 0-37 corrasponding with the one in labels.txt then use the saved number to call on the corrasponding wiki link and print it. I did the same with the classes labeled them all 0-37 and called on the corrasponding name and printed that as the name of the img. (this was done using plant.py)


## Running this project
# 1. Formating the data
1. Log into your nano
2. Make sure you have installed Resnet-18 and pytorch onto your nano
3. Download the raw dataset using this github link: https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color
4. Make sure to download it into the directory jetson-inference/python/training/classification/data
5. Go into the directory jetson-inference/python/training/classification
6. open the split_dataset.py file
7. go to the bottom of the file and replace the "a" in folder = "a" to the path of your downlodaded data (you can get the path by right clicking and clicking copypath)
![image](https://github.com/user-attachments/assets/8bb2d0e9-0981-4840-b8b6-1b0130d122cd)
8. navigate back to the jetson-inference/python/training/classification/data but this time cd into the name of your data folder (eg: cd 'data')
9. make a txt file in your data folder called labels.txt using the command 'mkdir labels.txt'
10. inside labels.txt write the name of all the classes in the dataset or copy from this doc: https://docs.google.com/document/d/e/2PACX-1vTQuOZXt6tJXMF1dMuq9tM68L1oTgFXbnBwqUUkldLkf92TljIUzM5cyIj_W_aaXTKvOyLmIw7kGrV1/pub
11. make sure you have 3 folders in your data folder named train, test, val and the labels.txt file
![image](https://github.com/user-attachments/assets/f89ab593-2da5-4f2f-a8c8-f96bf641bb0a)
# 2. Training the model
1. navigate back into jetson-inference
2. run the command echo 1 | sudo tee /proc/sys/vm/overcommit_memory (so that you dont get a error running the epoches)
3. run the docker contianer using this command ./docker/run.sh (it will ask for your password for the nano)
4. from inside the contianer change directories into jetson-inference/python/training/classification
5. here run the command python3 train.py --model-dir=models/color data/color --epochs=NumberOfEpochs ( replace NumberOfEpochs with the number of epoches you want to run)
  5a. this proccess will take serval hours you can stop anytime by press control+c and then resume with the command --resume and --epoch-start
  5b. incase you have named your dataset or model something other than color replace where it says color in that command to the name of your model and dataset
7. once the epoches finish running make sure you are in jetson-inference/python/training/classification
8. run the script python3 onnx_export.py --model-dir=models/color
   8a. incase you have named your model something other than color replace the color in that command with the name of the model folder
9.  look in jetson-inference/python/training/classification/models/color to make sure you have a file called resnet18.onnx
   9a. incase you have named your model something other than color replace the color in that command with the name of the model folder
![image](https://github.com/user-attachments/assets/b71684d7-029e-4499-ba07-e6f2a9a62a42)
10 exit docker by doing control + D
# 3. Running the algorithm
1. navigate from in your nano to jetson-inference/python/training/classification
2. download your img into that directory
   2a. the command to download the example model is curl -o y.jpg https://lh7-rt.googleusercontent.com/docsz/AD_4nXc2TGMslR7x-g1HCZtFBuhgHj7uVgWM9qR_qH4sDsBL9UrzzADEsJGfxr-pNlDuPjFYYY1UfvQkt71SnjE6mw5-P5vHD6JIdsAshMQSxNMPcmplirNPR6BLk4boTLvYs8tsrFpyus21BKupngYKhQ?  key=XPuR7jGeR8lFIqlXvdFmqw
3. go back to jetson-inference/python/training/classification\
4. run the command NET=models/color (without this the python script wont know where the model is)
  4a. incase you have named your model something other than color replace the color in that command with the name of the model folder
5. run the command DATASET=data/color (without this the python script wont know where the data is)
  4a. incase you have named your dataset something other than color replace the color in that command with the name of the data folder
6. finially run python3 plant.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt "photo" (replace photo with the name of your image plus .jpg or what it ends in)
7. Enjoy the output

**For running the live camera with the program run this command: imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt csi://0*

Required Libraries Needed
resnet-18
pytorch


##Video demonstration
https://youtu.be/xhp3iAaaAh0
