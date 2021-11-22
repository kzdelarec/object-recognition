import cv2
import numpy as np
import ssl
from urllib.request import urlopen, Request

# Primjeri

# cars
#url = 'https://img-ik.cars.co.za/news-site-za/images/2021/05/tr:n-news_large_crop/pics-2.jpg'
# people
#url = 'https://cdn0.gamesports.net/content_teasers/134000/134758_thumb.jpg?1634494749'
# sheep
#url='https://aldf.org/wp-content/uploads/2018/05/lamb-iStock-665494268-16x9-e1559777676675-1200x675.jpg'
# zebra
#url = 'https://www.passportandpixels.com/wp-content/uploads/2020/07/Serengeti-385_pp.jpg'
# bedroom
url = 'https://img.sunset02.com/sites/default/files/styles/4_3_horizontal_inbody_900x506/public/image/2016/08/main/luxurious-master-bedroom-sun-1114.jpg?itok=fBfiBQ0H'
# street
#url = 'https://static01.nyt.com/images/2020/06/22/opinion/20gillisWeb/merlin_173526783_192e498a-5699-411c-b1e5-64fa738eec17-articleLarge.jpg?quality=75&auto=webp&disable=upscale'
# fruits and vegetables
#url = 'https://blog-assets.shawacademy.com/uploads/2015/12/shutterstock_262781495.jpg'

# preuzimanje i otvaranje slike s interneta
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urlopen(url, context=ctx)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, -1)

# učivanje baze poznatih pobjekata
classNames: list = []
classFile = "coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#konfiguracija
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#dohvat detektiranih objekata
classIds, confs, bbox = net.detect(image, confThreshold=0.50)

#oznacavanje detektiranih objekata
for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    cv2.rectangle(image, box, (0, 0, 255), 1)
    cv2.putText(image, classNames[classId - 1] + '-' + str(round(confidence * 100)) + "%", (box[0] + 10, box[1] + 10),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("Result", image)
cv2.waitKey(0)
