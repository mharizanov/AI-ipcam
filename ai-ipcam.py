#!/usr/bin/env python
from darkflow.net.build import TFNet
import cv2
import os
import random
from subprocess import Popen
import time
from PIL import Image,ImageDraw
import numpy as np
import paho.mqtt.client as mqtt
import argparse
#In case Raspberry Pi camera is used instead of RTSP stream
import picamera
usepicamera = True

#RTSP captured frame frame_filename; preferably on RAM drive
#On Mac
#diskutil erasevolume HFS+ 'RAM Disk' `hdiutil attach -nomount ram://20480`
#frame_filename = '/Volumes/RAM Disk/frame'+str(random.randint(1,99999))+'.jpeg'
#On Raspberry Pi
#sudo mkdir /tmp/ramdisk; sudo chmod 777 /tmp/ramdisk
#sudo mount -t tmpfs -o size=16M tmpfs /tmp/ramdisk/

frame_filename = '/tmp/ramdisk/frame'+str(random.randint(1,99999))+'.jpeg'

#threshold parameter is below, keeping it too low will result in recognition errors
options = {"model": "cfg/tiny-yolo.cfg", "load": "tiny-yolo.weights",  "threshold": 0.55}

parser=argparse.ArgumentParser()
parser.add_argument(
  "--watch",  # name on the parser - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=['person', 'cat', 'dog', 'bird'],  # default if nothing is provided
)

parser.add_argument(
  "--stream",  # name on the parser - drop the `--` for positional/required parameters
  type=str,
  default= 'rtsp://192.168.1.153:554/onvif1', # default if nothing is provided
)

parser.add_argument(
  "--broker",  # name on the parser - drop the `--` for positional/required parameters
  type=str,
  default= '', # default if nothing is provided
)

parser.add_argument(
  "--topic",  # name on the parser - drop the `--` for positional/required parameters
  type=str,
  default= '', # default if nothing is provided
)

parser.add_argument(
  "--showimage",  # name on the parser - drop the `--` for positional/required parameters
  type=str,
  default='no', # default if nothing is provided
)

# parse the command line
args = parser.parse_args()
watch_list=args.watch
rtsp_stream=args.stream
broker_address=args.broker
mqtt_topic=args.topic
showimageflag=args.showimage

print("Watching for: %r" % watch_list)
print("Stream: %r" % rtsp_stream)
print("MQTT broker: %r" % broker_address)
print("MQTT topic: %r" % mqtt_topic)
print("Show image: %r" % showimageflag)

if usepicamera:
    camera = picamera.PiCamera()
    camera.resolution = (1024, 768)
else:
    #start a ffmpeg process that captures one frame every 5 seconds
    p = Popen(['ffmpeg', '-loglevel', 'panic', '-rtsp_transport', 'udp', '-i', rtsp_stream, '-f' ,'image2' ,'-pix_fmt', 'yuvj420p', '-r', '1/5' ,'-updatefirst', '1', frame_filename])

if broker_address!='':
    client = mqtt.Client("cameraclient_"+str(random.randint(1,99999999))) #create new instance
    client.connect(broker_address)


tfnet = TFNet(options)

while True:
    try:
        if usepicamera:
            camera.capture( frame_filename )
        curr_img = Image.open( frame_filename )
        curr_img_cv2 = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR) #is the frame good and can be opened?
        os.remove(frame_filename) #delete frame once it is processed, so we don't reprocess the same frame over
    except: # ..frame not ready, just snooze for a bit
        time.sleep(1)
        continue

    result = tfnet.return_predict(curr_img_cv2)
    print(result)

    if broker_address!='':
        client.publish(mqtt_topic, "".join([str(x) for x in result]) ) #publish

    saveflag=False
    namestr=''
    draw = ImageDraw.Draw(curr_img)
    for det_object in result:
        #if any(det_object['label'] in s for s in watch_list):
            draw.rectangle([det_object['topleft']['x'], det_object['topleft']['y'],det_object['bottomright']['x'], det_object['bottomright']['y']], outline=(255, 255, 0))
            draw.text([det_object['topleft']['x'], det_object['topleft']['y'] - 13], det_object['label']+' - ' + str(  "{0:.0f}%".format(det_object['confidence'] * 100) ) , fill=(255, 255, 0))
            saveflag=True
            namestr+='_'+str(det_object['label'])
    if saveflag == True:
        #curr_img.save('images/'+str(int(time.time()))+namestr+'.jpg')
        saveflag=False
    if showimageflag!='no':
        curr_img_cv2=cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("Security Feed", curr_img_cv2)
        if cv2.waitKey(50) & 0xFF == ord('q'): # wait for image render
            break
            continue
    time.sleep(1)
p.terminate()
cv2.destroyAllWindows() 
