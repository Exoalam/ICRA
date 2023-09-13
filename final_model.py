from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator
import pyrealsense2 as rs
from realsense_depth import *
from scipy.spatial import distance
from keras.applications import VGG19
from keras.layers import Dense, Flatten, Input,Concatenate
from keras.models import Model,load_model
from keras.preprocessing import image as keras_image
from keras.applications.vgg19 import preprocess_input


def orientation(image):
    resized_image = cv2.resize(image, (224, 224))  
    img_array = keras_image.img_to_array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(3, activation='linear')(x) 
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.load_weights('vgg19_1.h5')
    return model.predict(img_array)

def hybride_reg(x1,x2,x3,x4):
    input_linear = Input(shape=(1,), name="linear_input")
    input_poly = Input(shape=(3,), name="poly_input")
    linear_output = Dense(2, activation='linear', name="linear_output")(input_linear)
    poly_hidden = Dense(64, activation='relu')(input_poly)
    poly_hidden2 = Dense(32, activation='relu')(poly_hidden)
    poly_output = Dense(2, activation='linear', name="poly_output")(poly_hidden2)
    combined_output = Concatenate()([linear_output, poly_output])
    final_output = Dense(2, activation='linear')(combined_output)
    model = Model(inputs=[input_linear, input_poly], outputs=final_output)
    model.compile(optimizer='adam', loss='mse')
    model.load_weights('reg_new.h5')
    xd1 = np.array([[x1]])
    xd2 = np.array([[x2,x3,x4]])
    return model.predict([xd1,xd2])

custom_dtype = np.dtype([
    ('hit', np.int8),       
    ('accuracy', np.int8),
    ('class', np.int8)       
])
map = np.zeros((1000, 1000, 1000), dtype=custom_dtype)
pub_string = ""
def max_hit(points):
    final_list = []
    dis = 0
    for p in points:
        point_list = []
        value_list = []
        for point in points:
            dis = distance.euclidean(point, p)
            if dis < 10:
                point_list.append(point)
                value_list.append(map[point]['hit'])
        max_point = point_list[value_list.index(max(value_list))]
        if max_point not in final_list:        
            final_list.append(max_point)
    return final_list        


def reverse(depth_info,x,y,depth):
    depth_intrinsics = depth_info.profile.as_video_stream_profile().intrinsics
    pixel = rs.rs2_project_point_to_pixel(depth_intrinsics, [x,y,depth])
    return pixel

ori = None 
model = YOLO('yolov8n.pt')
dc = DepthCamera()
hit_map = np.zeros((1000,1000))
detect_list = [39]
while True:
    
    ret, depth_frame, color_frame, depth_info = dc.get_frame()
    img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    img_shape = img.shape
    results = model.predict(img)
    for r in results:     
        annotator = Annotator(color_frame)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0] 
            c = box.cls
            b1 = box.xyxy[0].detach().cpu().numpy()
            top_left = (int(b1[0]), int(b1[1]))
            bottom_right = (int(b1[2]), int(b1[3]))            
            x = box.xywh[0][0].detach().cpu().numpy() 
            y = box.xywh[0][1].detach().cpu().numpy()
            w = box.xywh[0][2].detach().cpu().numpy()
            h = box.xywh[0][3].detach().cpu().numpy()      
            point = int(x), int(y)
            cropped = color_frame[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
            if c in detect_list:
                #print(orientation(cropped))
                points = dc.Global_points(point[0],point[1])
                dx = 500+points[0][0]*100
                dy = 500+points[0][1]*100
                map[round(dx),round(dy),round(points[0][2]*100)]['hit'] += 1
                map[round(dx),round(dy),round(points[0][2]*100)]['class'] = c
                cv2.circle(color_frame, point, 4, (0, 0, 255))
                annotator.box_label(b, model.names[int(c)]+" x:"+str(round(points[0][0],2))+" y:"+str(round(points[0][1],2))+" z:"+str(round(points[0][2],2)))
                
    color_frame = annotator.result()  
    cv2.imshow('Vision', color_frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        ori = orientation(cropped)
        break  


indices = np.where(map['hit'] > 1)

points = list(zip(indices[0], indices[1], indices[2]))
points = max_hit(points)
coordinate = points[0]

#2nd Camera

ret, depth_frame, color_frame, depth_info = dc.get_frame()
x1 = float(coordinate[0])-500
x2 = float(coordinate[1])-500
points = reverse(depth_info,x1,x2,coordinate[2])
distance = np.sqrt(np.power(x1,2)+np.power(x2,2)+np.power(coordinate[2],2))
hw = hybride_reg(distance,ori[0][0],ori[0][1],ori[0][2])
cv2.circle(color_frame, (round(points[0]),round(points[1])), 4, (0, 0, 255))
w = round(np.abs(hw[0][0]))
h = round(np.abs(hw[0][1]))
x1 = round(points[0])
y1 = round(points[1])
x = int(x1 - w/2)
y = int(y1 - h/2)
x2 = int(x1 + w/2)
y2 = int(y1 + h/2)
cv2.rectangle(color_frame, (x, y), (x2, y2), (0, 255, 0), 2)
cv2.imshow('Vision', color_frame)     
cv2.waitKey(0)
cv2.destroyAllWindows()

