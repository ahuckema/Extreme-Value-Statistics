import numpy as np
from scipy.optimize import minimize
from scipy.stats import genextreme
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

csv_file_path = 'extracted_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
print(df.head())

j=1
AllCitiesMaxList=[]
while j<12:

    DataList=np.array(df.iloc[:,j])
    MaxList=[]
    i=0
    while i <= len(DataList):
        Out=max(DataList[i:i+365]) ###'alle anderen Jahre'
        if i == 365*36:
            Out=max(DataList[i:]) ##letze Jahr
        MaxList.append(Out)
        i=i+365

    AllCitiesMaxList.append(MaxList)

    j=j+1


# Generate a list of distinct colors for each dataset
colors = plt.cm.viridis(np.linspace(0, 1, len(AllCitiesMaxList)))

# Loop through each dataset
for j, color in zip(range(len(AllCitiesMaxList)), colors):
    Data = AllCitiesMaxList[j]
    index = list(range(len(Data)))

    # Plot the list with a unique color for each dataset
    plt.plot(index, Data, marker='o', linestyle='', color=color, label=f'Dataset {j + 1}')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Title')

# Show the legend
plt.legend()

# Show the plot
plt.show()




#Gamma Schätzen mit Deckars-Dehan

def Deckars(k,Data):
    x=Data[len(Data)-k+1:]
    y=Data[len(Data)-k]
    if y==0:
        y=0.1
    r=1
    Out1=(1/k)*np.sum((np.log(x)-np.log(y))**r)
    r=2
    Out2=(1/k)*np.sum((np.log(x)-np.log(y))**r)
   
    if Out1==0:
        Out1=0.1
    if Out2==0:
        Out2=0.1
    return Out1+(1-0.5*(1-((Out1**2)/Out2)**2))


j=1
TotalDeckarsList=[]
while j<1:
    DeckarsplotOneCity=[]
    k=1
    DataList=np.array(df.iloc[:,j])
    sortedList=sorted(DataList)
    genulledSortedList=[x for x in sortedList if x != 0]
    while k<len(genulledSortedList):
        DeckarsplotOneCity.append(Deckars(k,genulledSortedList))
        k=k+1
    TotalDeckarsList.append(DeckarsplotOneCity)
    print(j)
    j=j+1

# Generate a list of distinct colors for each dataset
colors = plt.cm.viridis(np.linspace(0, 1, len(TotalDeckarsList)))

# Loop through each dataset
for j, color in zip(range(len(TotalDeckarsList)), colors):
    Data = TotalDeckarsList[j]
    index = list(range(len(Data)))

    # Plot the list with a unique color for each dataset
    plt.plot(index, Data, color=color, label=f'Dataset {j + 1}')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Title')

# Show the legend
plt.legend()

# Show the plot
plt.show()










#Maximum likelyhood
ListofParam=[]
i=0
while i <11:
    params=genextreme.fit(AllCitiesMaxList[i])
    ListofParam.append(params)
    i=i+1

ListofParam=np.array(ListofParam)


# Data preparation
i = 0
while i < len(AllCitiesMaxList):
    AllCitiesMaxList[i] = np.array(np.ravel(AllCitiesMaxList[i]))
    AllCitiesMaxList[i] = np.sort(AllCitiesMaxList[i])
    i = i + 1

distributions = [
    {'label': '1', 'dist': genextreme(*ListofParam[0])},
    {'label': '2', 'dist': genextreme(*ListofParam[1])},
    {'label': '3', 'dist': genextreme(*ListofParam[2])},
    {'label': '4', 'dist': genextreme(*ListofParam[3])},
    {'label': '5', 'dist': genextreme(*ListofParam[4])},
    {'label': '6', 'dist': genextreme(*ListofParam[5])},
    {'label': '7', 'dist': genextreme(*ListofParam[6])},
    {'label': '8', 'dist': genextreme(*ListofParam[7])},
    {'label': '9', 'dist': genextreme(*ListofParam[8])},
    {'label': '10', 'dist': genextreme(*ListofParam[9])},
    {'label': '11', 'dist': genextreme(*ListofParam[10])},
]

# Create a figure with multiple subplots, specifying equal width and height
fig, axis = plt.subplots(4, 3)
k=0
l=0
for i, distribution in enumerate(distributions):

    if i==3:
        i=0
    if i==4:
        i=1
    if i==5:
        i=2
    if i==6:
        i=2
    if i==7:
        i=0
    if i==8:
        i=1
    if i==9:
        i=2
    if i==10:
        i=0
    
    # Calculate theoretical quantiles for the current distribution
    theoretical_quantiles = distribution['dist'].ppf(np.linspace(0.01, 0.99, len(AllCitiesMaxList[l])))

    # Plot the data against theoretical quantiles
    axis[k,i].scatter(AllCitiesMaxList[l], theoretical_quantiles, label=f'{distribution["label"]} Distribution')
    axis[k,i].set_ylabel('Theoretical Quantiles')
    axis[k,i].legend()

    # Add a diagonal line for reference
    axis[k,i].plot([min(AllCitiesMaxList[l]), max(AllCitiesMaxList[l])], [min(AllCitiesMaxList[l]), max(AllCitiesMaxList[l])], color='red', linestyle='--')

    l=l+1
    if l==3 or l==6 or l==9:
        k=k+1
# Set labels and title for the last subplot

# Show the plot
plt.tight_layout()
plt.show()


i=0
X_hat_List=[]
while i<len(AllCitiesMaxList):
    for j in [0,1,2]:
        if ListofParam[i][j]==0:
            ListofParam[i][j]=0.01
    pVec=np.linspace(0.001, 0.1, 100)
    #AllCitiesMaxList[i]=[x for x in AllCitiesMaxList[i] if x != 0]
    First=[1 for x in pVec]
    Second=[ListofParam[i][1] for x in pVec]
    
    x_hat=Second-(ListofParam[i][2]/ListofParam[i][0])*(First-(-np.log(1-pVec))**(-ListofParam[i][0]))
    X_hat_List.append(x_hat)
    i=i+1
###CONTINUE
colors = plt.cm.viridis(np.linspace(0, 1, len(X_hat_List)))

# Loop through each dataset
for j, color in zip(range(len(X_hat_List)), colors):
    Data = X_hat_List[j]
    index = pVec

    # Plot the list with a unique color for each dataset
    plt.plot(index,Data, color=color, label=f'Dataset {j + 1}')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Title')

# Show the legend
plt.legend()

# Show the plot
plt.show()