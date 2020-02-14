from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif , VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

df = pd.read_csv('C:\\Users\\joao-\\Desktop\\Python\\Datasets\\winequality-red.csv')
x= df.drop(['quality'],axis=1)
y = df.quality
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)
clf = RandomForestClassifier(n_estimators=420)

#Hybrid Method
start_time = time.time()
rfe = RFECV(clf, min_features_to_select = 3, step = 1 , cv=2, scoring='accuracy',n_jobs = -1)
fit_rfe = rfe.fit(x_train,y_train)
rfe_cols = x_train.columns[fit_rfe.support_]
print(f' Tempo de execução RFECV : {round(time.time() - start_time,2)} segundos')

''' Funcionamento -  Treina o modelo com todas as features, usa o atributo *feature importance* fornecido pelo próprio modelo
e elimina a feature menos importante e testa se o score melhorou com a mudança.
    Vantagens - Alta Performance e maior robustez e flexibilidade para lidar com muitas features.
    Desvantagem - 
    '''
#Wrapped Methods
start_time = time.time()
sfs = SequentialFeatureSelector(clf, k_features=(3,11), forward=True, floating=False,scoring='accuracy',cv=2, n_jobs=-1)
sfs = sfs.fit(x_train, y_train)
sfs_cols = x_train.columns[list(sfs.k_feature_idx_)]
print(f' Tempo de execução Foward Selection : {round(time.time() - start_time,2)} segundos')

start_time = time.time()
sbs = SequentialFeatureSelector(clf, k_features=(3,11), forward=False, floating=False,scoring='accuracy',cv=2, n_jobs=-1)
sbs = sfs.fit(x_train, y_train)
sbs_cols = x_train.columns[list(sbs.k_feature_idx_)]
print(f' Tempo de execução Backward Selection : {round(time.time() - start_time,2)} segundos')

''' Funcionamento Step Foward Selection - Treina o modelo com cada feature individualmente e seleciona a que obtiver o melhor score,
em seguida testa a melhor feature com todas as outras restantes escolhendo a dupla com melhor score e assim sucessivamente.

    Funcionamento Step Backward Selection - Treina o modelo com todas as features, em seguida remove de uma em uma e seleciona o conjunto com o melhor score,
    repete o processo eliminando uma feature de cada vez, ao invés de adicionar como no Step Foward.

    Vantagens - Detecta a interação entre as variaveis e geralmente acha o melhor conjunto de features para o modelo
    Desvantagem - No caso do SFS, ao fixar uma feature ela não pode ser descartada caso venha a atrapalhar o modelo quando avaliada junto a outras. Esse problema também afeta
o SBS analogamente.

    Vale ressaltar a existência do Exhaustive Feature Selection, que é a versão mais genérica do Wrapper Method,
    onde todas as possiveis combinações de features são avaliadas em troca de um enorme custo computacional.

    '''
#Filter Methods
start_time = time.time()
mutual_inf = SelectKBest(mutual_info_classif, k=8).fit(x_train, y_train)
mutual_inf_cols  = x_train.columns[mutual_inf.get_support()]
print(f' Tempo de execução Mutual Info : {round(time.time() - start_time,2)} segundos')

''' Funcionamento - Mutual Information é um método estatistico que calcula o quanto uma feature muda tendo em vista outra,
uma espécie de correlação mais ampla e genérica. Para fins de Machine Learning, Mutual Information mede o quanto a presença de certa feature
contribui para a previsão do alvo.
    Vantagem - Baixo custo computacional, suporte estatistico
    Desvantagem - Não é possivel determinar um número ótimo de features, este precisa ser arbitrado.

'''


start_time = time.time()
VT = VarianceThreshold(0.1)
VT.fit_transform(x_train)
VT_cols = x_train.columns[VT.get_support()]
print(f' Tempo de execução Variance Threshold : {round(time.time() - start_time,2)} segundos')

''' Funcionamento - Funciona como um filtro inicial que remove todas as features com variância menor do que o limite imposto
    Vantagem - Bom para um filtro inicial com baixissimo custo computacional
    Desvantagem - Não leva em consideração nenhum modelo específico de Machine Learning nem a relação entre as features

'''
#Embedded Method
start_time = time.time()
clf = clf.fit(x_train,y_train)
importances  = clf.feature_importances_
auxdf = pd.DataFrame({"Features": x_train.columns, "Importances":importances})
auxdf.sort_values('Importances',inplace=True)
embedded_cols = auxdf[3:].Features.tolist()
print(f' Tempo de execução Embedded Method : {round(time.time() - start_time,2)} segundos')
''' Funcionamento - Utiliza o atributo feature importance fornecido por modelos de arvore e seleciona as n features mais importantes
    Vantagens - Leva em consideração o modelo usado e a interação entre as features, além de ser menos propenso a overfitting
    Desvantagem - Não determina um número ótimo de features para o modelo, este precisa ser arbitrado


'''
clf0 = RandomForestClassifier(n_estimators=420)
clf1 = RandomForestClassifier(n_estimators=420)
clf2 = RandomForestClassifier(n_estimators=420)
clf3 = RandomForestClassifier(n_estimators=420)
clf4 = RandomForestClassifier(n_estimators=420)
clf5 = RandomForestClassifier(n_estimators=420)
clf6 = RandomForestClassifier(n_estimators=420)

clf0 = clf0.fit(x_train,y_train)
clf1 = clf1.fit(x_train[VT_cols],y_train)
clf2 = clf2.fit(x_train[mutual_inf_cols],y_train)
clf3 = clf3.fit(x_train[sfs_cols],y_train)
clf4 = clf4.fit(x_train[sbs_cols],y_train)
clf5 = clf5.fit(x_train[embedded_cols],y_train)
clf6 = clf6.fit(x_train[rfe_cols],y_train)


print(f'Baseline Accuracy : {accuracy_score(y_test,clf0.predict(x_test))}')
print(f'Variance Threshold Accuracy : {accuracy_score(y_test,clf1.predict(x_test[VT_cols]))}')
print(f'Mutual Information Accuracy : {accuracy_score(y_test,clf2.predict(x_test[mutual_inf_cols]))}')
print(f'Step Foward Accuracy : {accuracy_score(y_test,clf3.predict(x_test[sfs_cols]))}')
print(f'Step Backward Accuracy : {accuracy_score(y_test,clf4.predict(x_test[sbs_cols]))}')
print(f'Embedded Accuracy : {accuracy_score(y_test,clf5.predict(x_test[embedded_cols]))}')
print(f'RFECV Accuracy : {accuracy_score(y_test,clf6.predict(x_test[rfe_cols]))}')







    

