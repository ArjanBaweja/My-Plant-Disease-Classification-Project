#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# load the recognition network
net = imageNet(args.network, sys.argv)

# note: to hard-code the paths to load a model, the following API can be used:
#
# net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", 
#                 input_blob="input_0", output_blob="output_0")

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
font = cudaFont()

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # classify the image and get the topK predictions
    # if you only want the top class, you can simply run:
    #   class_id, confidence = net.Classify(img)
    predictions = net.Classify(img, topK=args.topK)

    plant_diseases = {
        0: "https://en.wikipedia.org/wiki/Apple_scab",
        1: "https://en.wikipedia.org/wiki/Black_rot",
        2: "https://en.wikipedia.org/wiki/Cedar_apple_rust",
        3: "https://en.wikipedia.org/wiki/Apple",
        4: "https://en.wikipedia.org/wiki/Blueberry",
        5: "https://en.wikipedia.org/wiki/Cherry",
        6: "https://en.wikipedia.org/wiki/Powdery_mildew",
        7: "https://en.wikipedia.org/wiki/Cercospora",
        8: "https://en.wikipedia.org/wiki/Rust_(fungus)",
        9: "https://en.wikipedia.org/wiki/Maize",
        10: "https://en.wikipedia.org/wiki/Northern_Leaf_Blight",
        11: "https://en.wikipedia.org/wiki/Black_rot",
        12: "https://en.wikipedia.org/wiki/Esca_(Black_Measles)",
        13: "https://en.wikipedia.org/wiki/Isariopsis_Leaf_Spot",
        14: "https://en.wikipedia.org/wiki/Grape",
        15: "https://en.wikipedia.org/wiki/Huanglongbing",
        16: "https://en.wikipedia.org/wiki/Bacterial_spot",
        17: "https://en.wikipedia.org/wiki/Peach",
        18: "https://en.wikipedia.org/wiki/Bacterial_spot",
        19: "https://en.wikipedia.org/wiki/Bell_pepper",
        20: "https://en.wikipedia.org/wiki/Potato",
        21: "https://en.wikipedia.org/wiki/Early_blight",
        22: "https://en.wikipedia.org/wiki/Potato",
        23: "https://en.wikipedia.org/wiki/Late_blight",
        24: "https://en.wikipedia.org/wiki/Raspberry",
        25: "https://en.wikipedia.org/wiki/Soybean",
        26: "https://en.wikipedia.org/wiki/Powdery_mildew",
        27: "https://en.wikipedia.org/wiki/Strawberry",
        28: "https://en.wikipedia.org/wiki/Leaf_scorch",
        29: "https://en.wikipedia.org/wiki/Bacterial_spot",
        30: "https://en.wikipedia.org/wiki/Early_blight",
        31: "https://en.wikipedia.org/wiki/Tomato",
        32: "https://en.wikipedia.org/wiki/Late_blight",
        33: "https://en.wikipedia.org/wiki/Leaf_Mold",
        34: "https://en.wikipedia.org/wiki/Septoria_leaf_spot",
        35: "https://en.wikipedia.org/wiki/Spider_mite",
        36: "https://en.wikipedia.org/wiki/Target_Spot",
        37: "https://en.wikipedia.org/wiki/Tomato_mosaic_virus",
        38: "https://en.wikipedia.org/wiki/Tomato_Yellow_Leaf_Curl_Virus"
}
    plant_disease_dict = {
        0: "Apple___Apple_scab",
        1: "Apple___Black_rot",
        2: "Apple___Cedar_apple_rust",
        3: "Apple___healthy",
        4: "Blueberry___healthy",
        5: "Cherry_(including_sour)___healthy", 
        6: "Cherry_(including_sour)___Powdery_mildew",
        7: "Corn_(maize)___Cercospora_leaf_spotGray_leaf_spot",
        8: "Corn_(maize)___Common_rust_",
        9: "Corn_(maize)___healthy",
        10: "Corn_(maize)___Northern_Leaf_Blight",
        11: "Grape___Black_rot",
        12: "Grape___Esca_(Black_Measles)",
        13: "Grape___healthy",
        14: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        15: "Orange___Haunglongbing_(Citrus_greening)",
        16: "Peach___Bacterial_spot",
        17: "Peach___healthy",
        18: "Pepper,_bell___Bacterial_spot",
        29: "Pepper,_bell___healthy",
        20: "Potato___Early_blight",
        21: "Potato___healthy",  
        22: "Potato___Late_blight",
        23: "Raspberry___healthy",
        24: "Soybean___healthy",
        25: "Squash___Powdery_mildew",
        26: "Strawberry___healthy",
        27: "Strawberry___Leaf_scorch",
        28: "Tomato___Bacterial_spot",
        29: "Tomato___Early_blight",
        30: "Tomato___healthy", 
        31: "Tomato___Late_blight",
        32: "Tomato___Leaf_Mold",
        33: "Tomato___Septoria_leaf_spot",
        34: "Tomato___Spider_mitesTwo-spotted_spider_mite", 
        35: "Tomato___Target_Spot",
        36: "Tomato___Tomato_mosaic_virus",
        37: "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}
    # draw predicted class labels
    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID)
        confidence *= 100.0


        print(f"imagenet: {confidence:05.2f}% class #{classID} {plant_disease_dict[classID]}")
        print(f"more resources for this plant/diseases here: {plant_diseases[classID]}" )
        


        font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", 
                         x=5, y=5 + n * (font.GetSize() + 5),
                         color=font.White, background=font.Gray40)
                         
    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break