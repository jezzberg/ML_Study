import pandas as pd


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

# dataset in stored in a DataFrame
df = pd.read_csv('data_vreme1.csv')

attributes = df.columns[0:4] #collection of attribute names
className = df.columns[-1] #name of column with class labels (Joc)

classLabels = set(df[className]) # the class labels are Da, no

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
testInstance = ['Soare', 'Mare', 'Normala', 'Absent']

# goal: determine probability values for the testInstance using Naive BaDa (see documentation) ... 

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


attribName = attributes[1]
ck_temperatura = aparitii(df[attribName],temp,list_joc)


attribName = attributes[2]
ck_umiditate = aparitii(df[attribName],temp,list_joc)

            
attribName = attributes[3]
ck_vant = aparitii(df[attribName],temp,list_joc)



element = temp.copy()

suma = 0
for c in ck_joc:
    if temp[c] == 0:
        suma += ck_joc[c]
    temp[c] = float(ck_joc[c])


rez = temp.copy() 


for c in rez:
    rez[c] = rez[c] / suma


laplace = input("Laplace Correction?")

if laplace == "Da":
    # Laplace
    index = 0
    for test in testInstance:
        if test in  ck_starea_vremii and index == 0:
            for c in rez: 
                elProd = (ck_starea_vremii[test][c] + 1)/(temp[c]+len(ck_joc))
                rez[c] = rez[c] * elProd
    
        if test in  ck_temperatura and index == 1:
            for c in rez:  
                elProd = (ck_temperatura[test][c] + 1)/(temp[c]+len(ck_joc))
                rez[c] = rez[c] * elProd
        
        if test in  ck_umiditate and index == 2:
            for c in rez:
                elProd = (ck_umiditate[test][c] + 1)/(temp[c]+len(ck_joc))
                rez[c] = rez[c] * elProd
        
        if test in  ck_vant and index == 3:
            for c in rez:   
                elProd = (ck_vant[test][c] + 1)/(temp[c]+len(ck_joc))
                rez[c] = rez[c] * elProd
        index +=1
else:
    index = 0
    for test in testInstance:
        if test in  ck_starea_vremii and index == 0:
            for c in rez: 
                elProd = ck_starea_vremii[test][c]/temp[c]
                rez[c] = rez[c] * elProd
    
        if test in  ck_temperatura and index == 1:
            for c in rez:  
                elProd = ck_temperatura[test][c]/temp[c]
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
if laplace == "Da":        
    for result in rez:
        print("Laplace result for "+result+" = "+str(round(rez[result]*1000 ,2))+" * 10^(-3)")
else:
    for result in rez:
        print("Laplace result for "+result+" = "+str(round(rez[result]*1000 ,2))+" * 10^(-3)")
        
        
if rez["Da"] > rez["Nu"]:
    print(str(testInstance)+" Da")
elif rez["Da"] < rez["Nu"]:
    print(str(testInstance)+" Nu")



















