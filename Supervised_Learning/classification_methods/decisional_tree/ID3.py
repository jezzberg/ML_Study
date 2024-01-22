import pandas as pd
from datasetprocessor import *
import math
import numpy as np

class Node:
    def __init__(self, name = 'root', branchName = '', children = []):
        self.name = name
        self.branchName = branchName
        self.children = children.copy()

def printNode(node, parentStrLen = 0):
    indent = ' ' * parentStrLen
    nodeStr = ''
    if node.branchName == '': #root node
        nodeStr = node.name
        print(indent, nodeStr)
    else:
        nodeStr = '--'+node.branchName+'--> '+node.name
        print(indent, nodeStr)

    for child in node.children:
        printNode(child, parentStrLen + len(nodeStr))

def ID3(ds, branchName, attribList):
    # branchName este numele ramurii dintre nodul curent si parintele sau
    # attribList este o lista ce contine numele atributelor 
    dp = DatasetProcessor(ds)	
    # print(ds)
    # print("\n")
    # print(attribList)
    node = Node()
    node.branchName = branchName
    print(attribList)
    print(branchName)
    # daca toate instantele din ds au aceeasi clasa, atunci
	# node.name = numele acelei clase
	# return node
    
   
    if 'Da' not in dp.labelCount:
        node.name = "Nu"
        print("nu")
        return node
    if 'Nu' not in dp.labelCount:
        node.name = "Da"
        print("da")
        return node
    
    

    # daca lista atributelor este goala, atunci
	# node.name = clasa care apare cel mai frecvent in ds
	# return node
    
    if not len(dp.attributes):
        node.name = dp.getLabelWithMaxCount()
        print("out")
        return node
    
    
    A = getMinEntropy(ds,attribList)
    node.name = A
    
    print(A)
    
    
    Avalues = dp.getAttribValues(A)# valorile posibile ale atributului A
    # print("Atribute: "+str(Avalues))
    
    for val in Avalues:
        subset = dp.getSubset(A, val) # submultimea lui ds care contine doar instantele cu valoarea val a atributului A
        # print(subset)
        if len(subset) == 0: # daca submultimea este goala
            node.children.append(Node(dp.getLabelWithMaxCount()) ) # un nou nod cu numele dat de clasa care apare cel mai frecvent in ds 
        else:
              
            newAttribList = attribList.copy() # o noua lista ce contine atributele din attribList, mai putin atributul A
            newAttribList.remove(A)
            # newAttribList = attribList[1:]
            node.children.append(ID3(subset, val, newAttribList)) # se apeleaza recursiv functia pentru generarea nodului descendent

    return node


def getEntropyAttr(ds, attribName):
    listVreme = {}
    temp = {}
    entropy= {}
    dp = DatasetProcessor(ds)
    
    print("-----------------------------------------------------")
    print(ds)    
    for c in ds[dp.className]:
        if c not in temp:
            temp[c] = 0    
        
    
    for vreme in ds[attribName]:
       listVreme[vreme] = temp.copy()
       
    
    
    for i in ds.index:
    # print(i)
        for label in dp.classLabels:
            # print(ds[dp.className][i])
            if label == ds[dp.className][i]:
                listVreme[ds[attribName][i]][label]+=1
                
    
   
    H = 0
    for vreme in dp.getAttribValues(attribName):
        # print(vreme)
        Hvreme = 0.0
        suma = 0
        for label in dp.classLabels:
            suma += listVreme[vreme][label]
            
            
        P =  suma / dp.instanceCount
        
        for label in dp.classLabels:
            if suma != 0:
                if listVreme[vreme][label]/suma != 0:
                    Hvreme -= listVreme[vreme][label]/suma * math.log(listVreme[vreme][label]/suma ,2)
           
                
        # print("H("+vreme+") = "+str(Hvreme))
        
        H += P*Hvreme
        # print(H)
    entropy[attribName] = H 
    return entropy


def getMinEntropy(ds, attribList):
    entropy = []
    
    for atrib in attribList:
        h = getEntropyAttr(ds, atrib)
        # print(h)
        entropy.append(h[atrib])
        
    entropy = np.array(entropy)
    return attribList[np.argmin(entropy)]


# ex 2

def tree_search(node, row):
    if node.children:
        searchInChild = row[node.name]
        for c in node.children:
            if c.branchName == searchInChild:
                return tree_search(c, row)
    else:
        return node


def classification(tree, testare_ds):
    dp = DatasetProcessor(testare_ds)	
    
    result = [] 
    wrong = 0
    subsetVreme4 = testare_ds[testare_ds.columns[-1]].values
    # print(subsetVreme4)
    for i, row in testare_ds.iterrows():
        
        node = tree_search(tree, row)
        
        result.append(node.name)
 
    print(result)
    for i in range(len(result)):
        if result[i] != subsetVreme4[i]:
            wrong += 1
            
            
    if wrong != 0 or len(result) != 0:
        error = wrong / len(result)
        return error
    return 0

if __name__ == '__main__':
    ds = pd.read_csv('data_vreme3.csv')
    dp = DatasetProcessor(ds)

    Alist = list(dp.attributes)
    root = ID3(ds, '', Alist)
    printNode(root)
    
    #   ex 2
    
    testare_ds = pd.read_csv('data_vreme4.csv')
    testare_dp = DatasetProcessor(ds)
    print("eroarea de clasificare este: "+str(classification(root, testare_ds)))
    


