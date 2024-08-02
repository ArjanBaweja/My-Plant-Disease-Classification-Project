# Plant Disease Classification Project

This project uses the Plant Village dataset and the NVIDIA Jetson Nano to run an image classification program that identifies photos of 38 classes of plants, indicating whether the plant is healthy or has a disease. It then provides a Wikipedia link to help you learn more and aid in caring for your plant.

Input img:
![0c620ec5-11cf-4120-94ab-1311e99df147___FREC_Scab 3131](https://github.com/user-attachments/assets/2546e04e-9033-4075-9ba5-be72e0d5d71d)

(Apple Leaf with Scab Diease)

Output:
![WhatsApp Image 2024-08-03 at 00 20 20_257ecb6a](https://github.com/user-attachments/assets/b83220a7-6c2b-4875-83ed-7a468df7eaa7)

(Correctly Identifyed as Apple__Scab)
(Wiki Link: https://en.wikipedia.org/wiki/Apple_scab)

This project can be used by farmers or just passionate gardeners, solving the problem of not knowing what to do when you see spots on your plants. With this project, you can learn what disease it is or if it’s just healthy and what further actions you can take to help your plant. The idea of providing a Wikipedia link is unique to my project. While there are other projects that can identify what plant disease it is, not many provide a solution or a next step.

## The Algorithm

The algorithm behind this project works in three steps.

The first step is downloading and formatting the dataset. For this project, I used the Plant Village dataset. This dataset has 38 classes of diseases on various plants, with each class having around 500-1000 images. However, the original Plant Village dataset is not formatted in the correct way for the next steps, because to train the AI, the dataset needs to be split into Train, Test, Val, and a labels.txt file which consists of the names of all 38 classes. The format needs to be split into these three classes because the AI needs to have a split of photos to train, use for accuracy, and to test. My data was split into an 80, 10, 10 split, with 80% of the data being in the train folder. (This was done using split_datasets.py)

The second step to this algorithm is training the model. To train the model, you have to get the AI to keep looking at the train images to try and learn by itself what type of image belongs in which class. The project uses ResNet-18 for this. Then the model uses the val photos to test its accuracy. One cycle of this is called an epoch. For my model, I ran 10 epochs; however, you can run as many as you want. I recommend running more than 25. (This was done using train.py)

Lastly, you can run the algorithm with any photo of your choice. The algorithm will try to fit the image you sent it into one of the classes and create a prediction. It will then print the prediction along with the name of the class, the % sure it is, and a link to Wikipedia. This works because the algorithm assigns every class a number from 0-37 based on its position in the labels.txt file in the dataset. Then whatever it predicts, it finds the number for that class and saves it. I then used this number to print the name of the class and a link to Wikipedia by creating two dictionaries: one with all links to Wikipedia labeled 0-37 corresponding with the one in labels.txt, then using the saved number to call on the corresponding wiki link and print it. I did the same with the classes, labeled them all 0-37, and called on the corresponding name and printed that as the name of the image. (This was done using plant.py)


## Running this project
# 1. Formating the data
1. Log into your Nano.
2. Make sure you have installed ResNet-18 and PyTorch onto your Nano.
3. Download the raw dataset using this github link: https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color.
4. Make sure to download it into the directory jetson-inference/python/training/classification/data.
5. Go into the directory jetson-inference/python/training/classification.
6. Open the split_dataset.py file.
7. Go to the bottom of the file and replace the "a" in folder = "a" to the path of your downloaded data (you can get the path by right-clicking and clicking "Copy Path").

![image](https://github.com/user-attachments/assets/8bb2d0e9-0981-4840-b8b6-1b0130d122cd)

9. Navigate back to jetson-inference/python/training/classification/data, but this time cd into the name of your data folder (e.g., cd 'data').
10. Make a txt file in your data folder called labels.txt using the command 'mkdir labels.txt'.
11. Inside labels.txt, write the name of all the classes in the dataset or copy from: https://docs.google.com/document/d/e/2PACX-1vTQuOZXt6tJXMF1dMuq9tM68L1oTgFXbnBwqUUkldLkf92TljIUzM5cyIj_W_aaXTKvOyLmIw7kGrV1/pub
12. Make sure you have three folders in your data folder named train, test, val, and the labels.txt file.

![image](https://github.com/user-attachments/assets/f89ab593-2da5-4f2f-a8c8-f96bf641bb0a)
# 2. Training the model
1. Navigate back into jetson-inference.
2. Run the command echo 1 | sudo tee /proc/sys/vm/overcommit_memory (so that you don’t get an error running the epochs).
3. Run the Docker container using this command ./docker/run.sh (it will ask for your password for the Nano).
4. From inside the container, change directories into jetson-inference/python/training/classification.
5. Here, run the command python3 train.py --model-dir=models/color data/color --epochs=NumberOfEpochs (replace NumberOfEpochs with the number of epochs you want to run).
  5a. This process will take several hours. You can stop anytime by pressing Ctrl+C and then resume with the --resume and --epoch-start options.
  5b. In case you have named your dataset or model something other than "color", replace "color" in that command with the name of your model and dataset.
6. Once the epochs finish running, make sure you are in jetson-inference/python/training/classification.
7. Run the script python3 onnx_export.py --model-dir=models/color.
   8a. In case you have named your model something other than "color", replace "color" in that command with the name of the model folder.
8.  Look in jetson-inference/python/training/classification/models/color to make sure you have a file called resnet18.onnx.
   8a. In case you have named your model something other than "color", replace "color" in that command with the name of the model folder.
    
![image](https://github.com/user-attachments/assets/b71684d7-029e-4499-ba07-e6f2a9a62a42)

9 Exit Docker by pressing Ctrl+D.
# 3. Running the algorithm
1. Navigate from your Nano to jetson-inference/python/training/classification.
2. Download your image into that directory.
   2a. The command to download the example image is: '-o y.jpg https://lh7-rt.googleusercontent.com/docsz/AD_4nXc2TGMslR7x-g1HCZtFBuhgHj7uVgWM9qR_qH4sDsBL9UrzzADEsJGfxr-pNlDuPjFYYY1UfvQkt71SnjE6mw5-P5vHD6JIdsAshMQSxNMPcmplirNPR6BLk4boTLvYs8tsrFpyus21BKupngYKhQ?  key=XPuR7jGeR8lFIqlXvdFmqw'.
3. Go back to jetson-inference/python/training/classification.
4. Set the model directory with the command NET=models/color (without this, the Python script won’t know where the model is).
  4a. In case you have named your model something other than "color", replace "color" in that command with the name of the model folder.
5. Set the dataset directory with the command DATASET=data/color (without this, the Python script won’t know where the data is).
  4a. In case you have named your dataset something other than "color", replace "color" in that command with the name of the data folder.
6. Finally, run python3 plant.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt "photo" (replace "photo" with the name of your image file including the extension).
7. Enjoy the output.

**For running the live camera with the program run this command: imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt csi://0*

Required Libraries Needed
resnet-18
pytorch


#Video demonstration
https://youtu.be/xhp3iAaaAh0
