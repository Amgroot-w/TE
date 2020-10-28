# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 00:40:42 2018

@author: Arevalo
"""

import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#constants
sense=7
TESize = 0.5
allTrust = []

#Current directory
cwd = os.getcwd() #to get current working directory


#Training files
d00TrFeTrn = np.loadtxt(cwd + "\\TE_process_dataset\\d00.dat", dtype= float)
d00TrFePl20 = np.transpose(d00TrFeTrn)
d00TrFe = d00TrFePl20[0:len(d00TrFePl20)-20]
d01TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d01.dat", dtype= float)
d02TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d02.dat", dtype= float)
d03TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d03.dat", dtype= float)
d04TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d04.dat", dtype= float)
d05TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d05.dat", dtype= float)
d06TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d06.dat", dtype= float)
d07TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d07.dat", dtype= float)
d08TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d08.dat", dtype= float)
d09TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d09.dat", dtype= float)
d10TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d10.dat", dtype= float)
d11TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d11.dat", dtype= float)
d12TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d12.dat", dtype= float)
d13TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d13.dat", dtype= float)
d14TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d14.dat", dtype= float)
d15TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d15.dat", dtype= float)
d16TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d16.dat", dtype= float)
d17TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d17.dat", dtype= float)
d18TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d18.dat", dtype= float)
d19TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d19.dat", dtype= float)
d20TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d20.dat", dtype= float)
d21TrFe = np.loadtxt(cwd + "\\TE_process_dataset\\d21.dat", dtype= float)

#Testing files
d00TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d00_te.dat", dtype= float)
d01TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d01_te.dat", dtype= float)
d02TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d02_te.dat", dtype= float)
d03TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d03_te.dat", dtype= float)
d04TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d04_te.dat", dtype= float)
d05TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d05_te.dat", dtype= float)
d06TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d06_te.dat", dtype= float)
d07TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d07_te.dat", dtype= float)
d08TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d08_te.dat", dtype= float)
d09TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d09_te.dat", dtype= float)
d10TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d10_te.dat", dtype= float)
d11TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d11_te.dat", dtype= float)
d12TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d12_te.dat", dtype= float)
d13TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d13_te.dat", dtype= float)
d14TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d14_te.dat", dtype= float)
d15TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d15_te.dat", dtype= float)
d16TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d16_te.dat", dtype= float)
d17TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d17_te.dat", dtype= float)
d18TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d18_te.dat", dtype= float)
d19TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d19_te.dat", dtype= float)
d20TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d20_te.dat", dtype= float)
d21TeFe = np.loadtxt(cwd + "\\TE_process_dataset\\d21_te.dat", dtype= float)


'''Multiclass classifiers'''

#Labels Arrays
d00TrLa = np.zeros([len(d00TrFe),1])
d01TrLa = np.ones([len(d01TrFe),1])*1
d02TrLa = np.ones([len(d02TrFe),1])*2
d03TrLa = np.ones([len(d03TrFe),1])*3
d04TrLa = np.ones([len(d04TrFe),1])*4
d05TrLa = np.ones([len(d05TrFe),1])*5
d06TrLa = np.ones([len(d06TrFe),1])*6
d07TrLa = np.ones([len(d07TrFe),1])*7
d08TrLa = np.ones([len(d08TrFe),1])*8
d09TrLa = np.ones([len(d09TrFe),1])*9
d10TrLa = np.ones([len(d10TrFe),1])*10
d11TrLa = np.ones([len(d11TrFe),1])*11
d12TrLa = np.ones([len(d12TrFe),1])*12
d13TrLa = np.ones([len(d13TrFe),1])*13
d14TrLa = np.ones([len(d14TrFe),1])*14
d15TrLa = np.ones([len(d15TrFe),1])*15
d16TrLa = np.ones([len(d16TrFe),1])*16
d17TrLa = np.ones([len(d17TrFe),1])*17
d18TrLa = np.ones([len(d18TrFe),1])*18
d19TrLa = np.ones([len(d19TrFe),1])*19
d20TrLa = np.ones([len(d20TrFe),1])*20
d21TrLa = np.ones([len(d21TrFe),1])*21

d00TeLa = np.zeros([len(d00TeFe),1])
d01TeLa = np.ones([len(d01TeFe),1])*1
d02TeLa = np.ones([len(d02TeFe),1])*2
d03TeLa = np.ones([len(d03TeFe),1])*3
d04TeLa = np.ones([len(d04TeFe),1])*4
d05TeLa = np.ones([len(d05TeFe),1])*5
d06TeLa = np.ones([len(d06TeFe),1])*6
d07TeLa = np.ones([len(d07TeFe),1])*7
d08TeLa = np.ones([len(d08TeFe),1])*8
d09TeLa = np.ones([len(d09TeFe),1])*9
d10TeLa = np.ones([len(d10TeFe),1])*10
d11TeLa = np.ones([len(d11TeFe),1])*11
d12TeLa = np.ones([len(d12TeFe),1])*12
d13TeLa = np.ones([len(d13TeFe),1])*13
d14TeLa = np.ones([len(d14TeFe),1])*14
d15TeLa = np.ones([len(d15TeFe),1])*15
d16TeLa = np.ones([len(d16TeFe),1])*16
d17TeLa = np.ones([len(d17TeFe),1])*17
d18TeLa = np.ones([len(d18TeFe),1])*18
d19TeLa = np.ones([len(d19TeFe),1])*19
d20TeLa = np.ones([len(d20TeFe),1])*20
d21TeLa = np.ones([len(d21TeFe),1])*21

#Train matrix Features and Labels
allTrFe = np.concatenate((d00TrFe,
                          d01TrFe,d02TrFe,d03TrFe,d04TrFe,d05TrFe,
                          d06TrFe,d07TrFe,d08TrFe,d09TrFe,d10TrFe,
                          d11TrFe,d12TrFe,d13TrFe,d14TrFe,d15TrFe,
                          d16TrFe,d17TrFe,d18TrFe,d19TrFe,d20TrFe,d21TrFe), axis=0)
allTrLa = np.concatenate((d00TrLa,
                          d01TrLa,d02TrLa,d03TrLa,d04TrLa,d05TrLa,
                          d06TrLa,d07TrLa,d08TrLa,d09TrLa,d10TrLa,
                          d11TrLa,d12TrLa,d13TrLa,d14TrLa,d15TrLa,
                          d16TrLa,d17TrLa,d18TrLa,d19TrLa,d20TrLa,d21TrLa), axis=0)

#Test matrix Features and Labels
allVaTeFe = np.concatenate((d00TeFe,
                          d01TeFe,d02TeFe,d03TeFe,d04TeFe,d05TeFe,
                          d06TeFe,d07TeFe,d08TeFe,d09TeFe,d10TeFe,
                          d11TeFe,d12TeFe,d13TeFe,d14TeFe,d15TeFe,
                          d16TeFe,d17TeFe,d18TeFe,d19TeFe,d20TeFe,d21TeFe), axis=0)
allVaTeLa = np.concatenate((d00TeLa,
                          d01TeLa,d02TeLa,d03TeLa,d04TeLa,d05TeLa,
                          d06TeLa,d07TeLa,d08TeLa,d09TeLa,d10TeLa,
                          d11TeLa,d12TeLa,d13TeLa,d14TeLa,d15TeLa,
                          d16TeLa,d17TeLa,d18TeLa,d19TeLa,d20TeLa,d21TeLa), axis=0)

##Split in validation and test data
allVaFe, allTeFe, allVaLa, allTeLa = train_test_split(allVaTeFe,allVaTeLa, test_size=TESize, random_state=10)


'''Binary'''

#Features Matrices
allTrFed01 = np.concatenate((  d00TrFe,  d01TrFe),  axis=0)
allTrFed02 = np.concatenate((  d00TrFe,  d02TrFe),  axis=0)
allTrFed03 = np.concatenate((  d00TrFe,  d03TrFe),  axis=0)
allTrFed04 = np.concatenate((  d00TrFe,  d04TrFe),  axis=0)
allTrFed05 = np.concatenate((  d00TrFe,  d05TrFe),  axis=0)
allTrFed06 = np.concatenate((  d00TrFe,  d06TrFe),  axis=0)
allTrFed07 = np.concatenate((  d00TrFe,  d07TrFe),  axis=0)
allTrFed08 = np.concatenate((  d00TrFe,  d08TrFe),  axis=0)
allTrFed09 = np.concatenate((  d00TrFe,  d09TrFe),  axis=0)
allTrFed10 = np.concatenate((  d00TrFe,  d10TrFe),  axis=0)
allTrFed11 = np.concatenate((  d00TrFe,  d11TrFe),  axis=0)
allTrFed12 = np.concatenate((  d00TrFe,  d12TrFe),  axis=0)
allTrFed13 = np.concatenate((  d00TrFe,  d13TrFe),  axis=0)
allTrFed14 = np.concatenate((  d00TrFe,  d14TrFe),  axis=0)
allTrFed15 = np.concatenate((  d00TrFe,  d15TrFe),  axis=0)
allTrFed16 = np.concatenate((  d00TrFe,  d16TrFe),  axis=0)
allTrFed17 = np.concatenate((  d00TrFe,  d17TrFe),  axis=0)
allTrFed18 = np.concatenate((  d00TrFe,  d18TrFe),  axis=0)
allTrFed19 = np.concatenate((  d00TrFe,  d19TrFe),  axis=0)
allTrFed20 = np.concatenate((  d00TrFe,  d20TrFe),  axis=0)
allTrFed21 = np.concatenate((  d00TrFe,  d21TrFe),  axis=0)

allVaTeFed01 = np.concatenate((  d00TeFe,  d01TeFe),  axis=0)
allVaTeFed02 = np.concatenate((  d00TeFe,  d02TeFe),  axis=0)
allVaTeFed03 = np.concatenate((  d00TeFe,  d03TeFe),  axis=0)
allVaTeFed04 = np.concatenate((  d00TeFe,  d04TeFe),  axis=0)
allVaTeFed05 = np.concatenate((  d00TeFe,  d05TeFe),  axis=0)
allVaTeFed06 = np.concatenate((  d00TeFe,  d06TeFe),  axis=0)
allVaTeFed07 = np.concatenate((  d00TeFe,  d07TeFe),  axis=0)
allVaTeFed08 = np.concatenate((  d00TeFe,  d08TeFe),  axis=0)
allVaTeFed09 = np.concatenate((  d00TeFe,  d09TeFe),  axis=0)
allVaTeFed10 = np.concatenate((  d00TeFe,  d10TeFe),  axis=0)
allVaTeFed11 = np.concatenate((  d00TeFe,  d11TeFe),  axis=0)
allVaTeFed12 = np.concatenate((  d00TeFe,  d12TeFe),  axis=0)
allVaTeFed13 = np.concatenate((  d00TeFe,  d13TeFe),  axis=0)
allVaTeFed14 = np.concatenate((  d00TeFe,  d14TeFe),  axis=0)
allVaTeFed15 = np.concatenate((  d00TeFe,  d15TeFe),  axis=0)
allVaTeFed16 = np.concatenate((  d00TeFe,  d16TeFe),  axis=0)
allVaTeFed17 = np.concatenate((  d00TeFe,  d17TeFe),  axis=0)
allVaTeFed18 = np.concatenate((  d00TeFe,  d18TeFe),  axis=0)
allVaTeFed19 = np.concatenate((  d00TeFe,  d19TeFe),  axis=0)
allVaTeFed20 = np.concatenate((  d00TeFe,  d20TeFe),  axis=0)
allVaTeFed21 = np.concatenate((  d00TeFe,  d21TeFe),  axis=0)

#Labels Matrices
dataSizeTr = len(d01TrFe)
allTrLaBin = np.concatenate( (   np.zeros([dataSizeTr,1]),   np.ones([dataSizeTr,1])   ), axis=0)

dataSizeTe = len(d01TeFe)
allVaTeLaBin = np.concatenate( (   np.zeros([dataSizeTe,1]),   np.ones([dataSizeTe,1])   ), axis=0)

#Split in validation and test data
allVaFed01, allTeFed01, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed01,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed02, allTeFed02, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed02,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed03, allTeFed03, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed03,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed04, allTeFed04, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed04,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed05, allTeFed05, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed05,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed06, allTeFed06, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed06,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed07, allTeFed07, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed07,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed08, allTeFed08, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed08,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed09, allTeFed09, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed09,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed10, allTeFed10, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed10,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed11, allTeFed11, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed11,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed12, allTeFed12, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed12,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed13, allTeFed13, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed13,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed14, allTeFed14, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed14,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed15, allTeFed15, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed15,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed16, allTeFed16, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed16,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed17, allTeFed17, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed17,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed18, allTeFed18, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed18,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed19, allTeFed19, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed19,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed20, allTeFed20, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed20,allVaTeLaBin, test_size=TESize, random_state=10)
allVaFed21, allTeFed21, allVaLaBin, allTeLaBin = train_test_split(allVaTeFed21,allVaTeLaBin, test_size=TESize, random_state=10)


#binary classifier
clf = KNeighborsClassifier(n_neighbors = 7)
clf.fit(d01TrFe,d01TrLa)
Y_pred = clf.predict(d01TeFe)    
cmbin = confusion_matrix(d01TeLa,Y_pred)
print("binary",cmbin)

#multiclass classifier
clf = KNeighborsClassifier(n_neighbors = 7)
clf.fit(allTrFe,allTrLa)
Y_pred = clf.predict(allTeFe)    
cmmulti = confusion_matrix(allTeLa,Y_pred)
print("multiclass",cmmulti)