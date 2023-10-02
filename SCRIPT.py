
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

#Carregamento dos dados

#Unidade 1
carregando_camera1 = np.load("C:\\Users\\daniel.carvalho\\OneDrive - Mob Serviços de Telecomunicações Ltda\\Área de Trabalho\\scenario1_seq_index_1-2422.npy")
dados_camera1 = carregando_camera1[:2000]
carregando_gps1 = np.load("C:\\Users\\daniel.carvalho\\OneDrive - Mob Serviços de Telecomunicações Ltda\\Área de Trabalho\\scenario1_unit1_loc_1-2422.npy")
dados_gps1 = carregando_gps1[:2000]
carregando_lidar1 = np.load("C:\\Users\\daniel.carvalho\\OneDrive - Mob Serviços de Telecomunicações Ltda\\Área de Trabalho\\scenario1_unit1_pwr_60ghz_1-2422.npy")
dados_lidar1 = carregando_lidar1[:50]
#Unidade 2
carregando_gps2 = np.load("C:\\Users\\daniel.carvalho\\OneDrive - Mob Serviços de Telecomunicações Ltda\\Área de Trabalho\\scenario1_unit2_loc_1-2422.npy")
dados_gps2 = carregando_gps2[:1600]


print(dados_camera1.shape)
print(dados_gps1.shape)
print(dados_lidar1.shape)
print(dados_gps2.shape)

k1 = np.reshape(dados_gps2,(-1 , 64))

print(k1.shape)

X = dados_lidar1
Y = k1

sc1 = StandardScaler()
sc1.fit(X)

X_norm = sc1.transform(X)

sc2 = StandardScaler()
sc2.fit(Y)

Y_norm = sc2.transform(Y)

print(X_norm)



X_train , X_test , Y_train , Y_test = train_test_split(X_norm , Y_norm , test_size=0.7)

rede = MLPRegressor(hidden_layer_sizes=(40,25,10) , 
                    max_iter=3000 , 
                    tol=0.000001, 
                    learning_rate_init=0.1 , 
                    solver="adam" , 
                    activation="tanh" , 
                    learning_rate="adaptive" , 
                    verbose=2
                   )

rede.fit(X_train , Y_train)

print(Y_norm)






