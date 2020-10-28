# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:41:25 2019

@author: Robot Hands & fernandoarevalo
"""
import warnings
#Turns off deprecation warnings associated with label.py
warnings.filterwarnings("ignore", category = DeprecationWarning)

import pickle
import numpy
import ParagraphOutput
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from opcua import Client
from opcua import ua

clientBGS = Client("opc.tcp://172.19.10.16:55105", timeout=4)

#loading in all the trained models
pickle_in1 = open('s1.pickle', 'rb')
Station1 = pickle.load(pickle_in1)


pickle_in2 = open('s2.pickle', 'rb')
predictor2 = pickle.load(pickle_in2)

pickle_in3 = open('s3.pickle', 'rb')
predictor3 = pickle.load(pickle_in3)

pickle_in4 = open('s4.pickle', 'rb')
predictor4 = pickle.load(pickle_in4)

def getVariables():
    variables = []    
    status = 0
    try:
        clientBGS.connect()
        listNodeStations = []

        listNodeStation1 = [
        ["1_Loading_BGLP.Software PLC_1.ProcessData_DB.Filling_Height_max.State", 6],
        ["1_Loading_BGLP.Software PLC_1.ProcessData_DB.Filling_Height_min.State", 6],
        ["1_Loading_BGLP.Software PLC_1.ProcessData_DB.Overflow.State", 6],
        ["1_Loading_BGLP.Software PLC_1.ProcessData_DB.Emergency", 6],
        ["1_Loading_BGLP.Software PLC_1.ProcessData_DB.Belt_Conveyor_ON/OFF", 6],
        ["1_Loading_BGLP.Software PLC_1.HealthSignal_DB.HealthCondition", 6]
        ]
        
        listNodeStation2 = [
        ["ET 200SP-Station_3.2_Storage_BGLP.ProcessData_DB.Filling_Height_max.State", 6],
        ["ET 200SP-Station_3.2_Storage_BGLP.ProcessData_DB.Filling_Height_min.State", 6],
        ["ET 200SP-Station_3.2_Storage_BGLP.ProcessData_DB.Overflow.State", 6],
        ["ET 200SP-Station_3.2_Storage_BGLP.ProcessData_DB.Emergency", 6],
        ["ET 200SP-Station_3.2_Storage_BGLP.ProcessData_DB.Pressure_Average", 6],
        ["ET 200SP-Station_3.2_Storage_BGLP.HealthSignal_DB.HealthCondition", 6]
        ]
        
        listNodeStation3 = [
        ["ET 200SP-Station_1.3_Weighing_BGLP.ProcessData_DB.Filling_Height_max.State", 6],
        ["ET 200SP-Station_1.3_Weighing_BGLP.ProcessData_DB.Filling_Height_min.State", 6],
        ["ET 200SP-Station_1.3_Weighing_BGLP.ProcessData_DB.Overflow.State", 6],
        ["ET 200SP-Station_1.3_Weighing_BGLP.ProcessData_DB.Emergency", 6],
        ["ET 200SP-Station_1.3_Weighing_BGLP.ProcessData_DB.Pressure_Average", 6],
        ["ET 200SP-Station_1.3_Weighing_BGLP.HealthSignal_DB.HealthCondition", 6]        
        ]
        
        listNodeStation4 = [
        ["ET 200SP-Station_4.4_Filling_BGLP.ProcessData_DB.Container_available.State", 6],
        ["ET 200SP-Station_4.4_Filling_BGLP.MachineCondition.RFID", 6],
        ["ET 200SP-Station_4.4_Filling_BGLP.ProcessData_DB.Emergency", 6],
        ["ET 200SP-Station_4.4_Filling_BGLP.MachineCondition.Pressure_Sensor", 6],
        ["ET 200SP-Station_4.4_Filling_BGLP.HealthSignal_DB.HealthCondition", 6]
        ]
            
        listNodeStations.append(listNodeStation1)
        listNodeStations.append(listNodeStation2)
        listNodeStations.append(listNodeStation3)
        listNodeStations.append(listNodeStation4)
           
        for x in listNodeStations:
            for y in x:
                variables.append(float(clientBGS.get_node(
                    ua.NodeId(y[0], int(y[1]))).get_value()))
        
        clientBGS.disconnect()
        
        """ Storing the variables"""
        fillingHeightMaxS1 = variables[0]
        fillingHeightMinS1 = variables[1]
        overflowS1         = variables[2]
        emergencyS1        = variables[3]
        motorSpeedS1       = variables[4]
        healthConditionS1  = variables[5]
        
        fillingHeightMaxS2 = variables[6]
        fillingHeightMinS2 = variables[7]
        overflowS2         = variables[8]
        emergencyS2        = variables[9]
        pressureAvgS2      = variables[10]
        healthConditionS2  = variables[11]
        
        fillingHeightMaxS3 = variables[12]
        fillingHeightMinS3 = variables[13]
        overflowS3         = variables[14]
        emergencyS3        = variables[15]
        pressureAvgS3      = variables[16]
        healthConditionS3  = variables[17]
        
        containerAvailS4   = variables[18]
        RFIDS4             = variables[19]
        emergencyS4        = variables[20]
        pressureAvgS4      = variables[21]
        healthConditionS4  = variables[22]
        
        status = 1
        return  fillingHeightMaxS1, fillingHeightMinS1, overflowS1, emergencyS1, motorSpeedS1, healthConditionS1, fillingHeightMaxS2, fillingHeightMinS2, overflowS2, emergencyS2, pressureAvgS2, healthConditionS2, fillingHeightMaxS3, fillingHeightMinS3, overflowS3, emergencyS3, pressureAvgS3, healthConditionS3, containerAvailS4, RFIDS4, emergencyS4, pressureAvgS4, healthConditionS4
    
    except Exception as e:
        print("Unable to retrieve variables")
        status = 0
        return status
  
fillingHeightMaxS1, fillingHeightMinS1, overflowS1, emergencyS1, motorSpeedS1, healthConditionS1,fillingHeightMaxS2, fillingHeightMinS2, overflowS2, emergencyS2, pressureAvgS2, healthConditionS2, fillingHeightMaxS3, fillingHeightMinS3, overflowS3, emergencyS3, pressureAvgS3, healthConditionS3, containerAvailS4, RFIDS4, emergencyS4, pressureAvgS4, healthConditionS4 = getVariables()

#call new values OPC-UA
getVariables()

S1Variables = [fillingHeightMaxS1, fillingHeightMinS1, overflowS1, emergencyS1, motorSpeedS1, healthConditionS1]
S2Variables = [fillingHeightMaxS2, fillingHeightMinS2, overflowS2, emergencyS2, pressureAvgS2, healthConditionS2]
S3Variables = [fillingHeightMaxS3, fillingHeightMinS3, overflowS3, emergencyS3, pressureAvgS3, healthConditionS3]
S4Variables = [containerAvailS4, RFIDS4, emergencyS4, pressureAvgS4, healthConditionS4]

#dummy values for testing @home
#x_names = ['S1 filling height max', 'S1 filling height min', 'S1 Overflow', 'S1 Emergency', 'S1 motorspeed']
#S1_Y_list = [1,1,1,1,1]
#x_values = numpy.array(x_names)

#Converts the string X labels into integers to feed to XGBoost
label_encoder = LabelEncoder()
S1_integers = label_encoder.fit_transform(S1Variables)
S2_integers = label_encoder.fit_transform(S2Variables)
S3_integers = label_encoder.fit_transform(S3Variables)
S4_integers = label_encoder.fit_transform(S4Variables)

onehot_encoder = OneHotEncoder(sparse = False)
integer1_encoded = S1_integers.reshape(len(S1_integers),1)
integer2_encoded = S2_integers.reshape(len(S2_integers),1)
integer3_encoded = S3_integers.reshape(len(S3_integers),1)
integer4_encoded = S4_integers.reshape(len(S4_integers),1)

S1_onehot = onehot_encoder.fit_transform(integer1_encoded)
S2_onehot = onehot_encoder.fit_transform(integer2_encoded)
S3_onehot = onehot_encoder.fit_transform(integer3_encoded)
S4_onehot = onehot_encoder.fit_transform(integer4_encoded)

#make a prediction
S1Prediction = Station1.score(S1_onehot, S1Variables)
S2Prediction = Station2.score(S2_onehot, S2Variables)
S3Prediction = Station3.score(S3_onehot, S3Variables)
S4Prediction = Station4.score(S4_onehot, S4Variables)
print(S1Prediction)
print(S2Prediction)
print(S3Prediction)
print(S4Prediction)
                    
#return a rule based on prediction
RuleBaseOutputS1(S1Prediction)
RuleBaseOutputS2(S2Prediction)
RuleBaseOutputS3(S3Prediction)
RuleBaseOutputS4(S4Prediction)