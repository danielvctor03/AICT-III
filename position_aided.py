import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#Unidade 1
dados_camera = np.load("C:\\Users\\daniel.carvalho\\OneDrive - Mob Serviços de Telecomunicações Ltda\\Área de Trabalho\\scenario1_seq_index_1-2422.npy")
dados_gps1 = np.load("C:\\Users\\daniel.carvalho\\OneDrive - Mob Serviços de Telecomunicações Ltda\\Área de Trabalho\\scenario1_unit1_loc_1-2422.npy")
dados_mmWave = np.load("C:\\Users\\daniel.carvalho\\OneDrive - Mob Serviços de Telecomunicações Ltda\\Área de Trabalho\\scenario1_unit1_pwr_60ghz_1-2422.npy")

#Unidade 2
dados_gps2 = np.load("C:\\Users\\daniel.carvalho\\OneDrive - Mob Serviços de Telecomunicações Ltda\\Área de Trabalho\\scenario1_unit2_loc_1-2422.npy")


print(dados_gps2)
print(dados_mmWave)

#Separação dos índices que mais ocorrem
dados_feixes = np.argmax(dados_mmWave , axis=1)
dados1 = np.column_stack((dados_gps2,dados_gps1))
def indices_n_maiores_valores_por_linha(matriz,n):
    indices_n_maiores_por_linha = []

    for linha in matriz:
        indices_ordenados = np.argsort(linha)[::-1]
        indices_n_maiores = indices_ordenados[:n]
        indices_n_maiores_por_linha.append(indices_n_maiores)

    return np.array(indices_n_maiores_por_linha)
top2=indices_n_maiores_valores_por_linha(dados_mmWave,2)
top3=indices_n_maiores_valores_por_linha(dados_mmWave,3)
top4=indices_n_maiores_valores_por_linha(dados_mmWave,4)
top5=indices_n_maiores_valores_por_linha(dados_mmWave,5)
print (dados1)
print (dados_feixes)
print (top2)
print (top3)
print (top4)
print (top5)

X=dados1
Y=top2
print (X.shape)
print (Y.shape)

sc1 = StandardScaler()
sc1.fit(X)

X_norm = sc1.transform(X)

X_train , X_test , Y_train , Y_test = train_test_split(X_norm , Y, test_size=0.7)

rede = MLPRegressor(hidden_layer_sizes=(500,300,100,60) , 
                    max_iter=3000 , 
                    tol=0.000001, 
                    learning_rate_init=0.1 , 
                    solver="adam" , 
                    activation="tanh" , 
                    learning_rate="adaptive" , 
                    verbose=True,
                    shuffle=True
                   )

rede.fit(X_train , Y_train)

Y_rede_pred = rede.predict(X_test)
Score_rede = r2_score(Y_test,Y_rede_pred)
print(Score_rede)
