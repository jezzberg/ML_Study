import pandas as pd
import math

# functions
def aparitii(atribut_,temp_,list_joc_):
    index = 0
    ck ={}
    for vremea in atribut_:
        if vremea not in ck:
            ck[vremea] = temp_.copy()
            
       
        ck[vremea][list_joc_[index]] += 1
        
        index+=1
    
    return ck

def dens_prob(x, miu, sigma):
    return math.exp((-(x-miu)**2)/(2*sigma**2))/math.sqrt(2*math.pi*(sigma**2))

# dataset in stored in a DataFrame
df = pd.read_csv('data_vreme2.csv')

attributes = df.columns[0:4] #collection of attribute names
className = df.columns[-1] #name of column with class labels (Joc)

classLabels = set(df[className]) # the class labels are Da, Nu

instanceCount = len(df)

#iterate through attributes
print('\nAttributes :')
for attribName in attributes:
    print(attribName)

#iterate through values of a certain attribute (a column of the dataset):
attribName = attributes[0] # Starea vremii
print('\nValues of', attribName, ':')
for val in df[attribName]:
    print(val)

#iterate through available class labels:
print('\nClasses : ')
for label in classLabels:
    print(label)

#iterate through class labels from data set (last column)
print('\nClass labels in dataset : ')
for label in df[className]:
    print(label)

# a new unclassified instance might look like this:
testInstance = ['Soare', 14, 'Normala', 'Absent']

# goal: determine probability values for the testInstance using Naive Bayes (see documentation) ... 

list_joc = []
ck_joc = {}
ck_starea_vremii = {} # {soare : {da : x, nu : y}, ..]
ck_temperatura = {}
ck_umiditate ={}
ck_vant = {}

temp = {}
rez = {}

for c in df[className]:
    if c not in ck_joc:
        ck_joc[c] = 0
        temp[c] = 0
    ck_joc[c] += 1
    list_joc.append(c)
    

ck_starea_vremii = aparitii(df[attribName],temp,list_joc)


# ex 2
attribName = attributes[1]
sigma_dict = temp.copy()
miu_dict = temp.copy()
index = 0

#[list_joc_[index]]
for c in df[attribName]:
    miu_dict[list_joc[index]] += c
    index +=1
        
for miu in miu_dict:
    miu_dict[miu] = float(miu_dict[miu]) / float(ck_joc[miu])
    
    print("µ "+miu+" = "+str(miu_dict[miu]))


index = 0
for c in df[attribName]:
    item = (c - miu_dict[list_joc[index]]) ** 2
    sigma_dict[list_joc[index]] += item
    index +=1

for sigma in sigma_dict:
    sigma_dict[sigma] = math.sqrt(float(sigma_dict[sigma]) / float(ck_joc[sigma]))
    
    print("σ "+sigma+" = "+str(round(sigma_dict[sigma],4)))

ck_temperatura = temp.copy()


#--------------------------------------------------------------------
attribName = attributes[2]
ck_umiditate = aparitii(df[attribName],temp,list_joc)

            
attribName = attributes[3]
ck_vant = aparitii(df[attribName],temp,list_joc)





suma = 0
for c in ck_joc:
    if temp[c] == 0:
        suma += ck_joc[c]
    temp[c] = float(ck_joc[c])


rez = temp.copy() 


for c in rez:
    rez[c] = rez[c] / suma

 

index = 0
for test in testInstance:
    if test in  ck_starea_vremii and index == 0:
        for c in rez: 
            elProd = ck_starea_vremii[test][c]/temp[c]
            rez[c] = rez[c] * elProd

    if  index == 1:
        for c in rez:  
            elProd = dens_prob(test, miu_dict[c], sigma_dict[c])
            print("P(Temp = " + str(test)+" | "+c +") = " + str(round(elProd,4)))
            rez[c] = rez[c] * elProd
    
    if test in  ck_umiditate and index == 2:
        for c in rez:
            elProd = ck_umiditate[test][c]/temp[c]
            rez[c] = rez[c] * elProd
    
    if test in  ck_vant and index == 3:
        for c in rez:   
            elProd = ck_vant[test][c]/temp[c]
            rez[c] = rez[c] * elProd
    index +=1


for result in rez:
    print("resultat pentru "+result+" = "+str(round(rez[result]*1000 ,2))+" * 10^(-3)")

if rez["Da"] > rez["Nu"]:
    print(str(testInstance)+" Da")
elif rez["Da"] < rez["Nu"]:
    print(str(testInstance)+" Nu")



















