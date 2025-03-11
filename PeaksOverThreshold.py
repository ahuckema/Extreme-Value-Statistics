import numpy as np
from scipy.optimize import minimize
from scipy.stats import genextreme, genpareto
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

csv_file_path = 'extracted_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
print(df.head())


i=1
AllExcessLists=[]
while i<12:
    DataList=np.array(df.iloc[:,i])
    U_List=np.linspace(min(DataList),max(DataList),100)
    meanExcessList=[]
    for u in U_List:
        CounterOverU=[1 for x in DataList if x>u]
        Summant=DataList-u
        Summant = [x if x > 0 else 0 for x in Summant]
        meanExcessValue=(1/np.sum(CounterOverU))*sum(Summant)
        meanExcessList.append(meanExcessValue)
        
    i=i+1
    AllExcessLists.append(meanExcessList)


# Generate a list of distinct colors for each dataset
colors = plt.cm.viridis(np.linspace(0, 1, len(AllExcessLists)))

# Loop through each dataset
for j, color in zip(range(len(AllExcessLists)), colors):
    Data = AllExcessLists[j]
    index = list(range(len(Data)))

    # Plot the list with a unique color for each dataset
    plt.plot(index, Data, marker='o', linestyle='-', color=color, label=f'Dataset {j + 1}')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Title')

# Show the legend
plt.legend()

# Show the plot
plt.show()



k_List=[None,20,50,21.25,31,35,33,20,30,25,28,25]
k_List=[None,28,20,20,20,20,20,20,20,20,20,20]
Data=[]
i=1
while i<12:
    DataList=np.array(df.iloc[:,i])
    Data.append([x-k_List[i] for x in DataList if x>k_List[i]])
    i=i+1

#Maximum likelyhood
ListofParam=[]
i=0
while i <11:
    params=genpareto.fit(Data[i])
    ListofParam.append(params)
    i=i+1

ListofParam=np.array(ListofParam)

AllCitiesMaxList=Data

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




ListofK = []
i = 0

for index, distribution in enumerate(distributions):
    ListOfMSE = []
    index = 1
    while index < 40:  # Use len(distributions) instead of 40


        Data=[]
        h=1
        while h<12:
            DataList=np.array(df.iloc[:,h])
            Data.append([x-index for x in DataList if x>index])
            h=h+1

        #Maximum likelyhood
        ListofParam=[]
        h=0
        while h <11:
            params=genpareto.fit(Data[h])
            ListofParam.append(params)
            h=h+1

        ListofParam=np.array(ListofParam)

        AllCitiesMaxList=Data

        # Data preparation
        h = 0
        while h < len(AllCitiesMaxList):
            AllCitiesMaxList[h] = np.array(np.ravel(AllCitiesMaxList[h]))
            AllCitiesMaxList[h] = np.sort(AllCitiesMaxList[h])
            h = h + 1

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



        theoretical_quantiles = distributions[i]['dist'].ppf(np.linspace(0.01, 0.99, len(Data[i])))
        DataList = Data[i]

        j = 0
        Sum = []
        while j < len(DataList):
            Out = (theoretical_quantiles[j] - DataList[j]) ** 2
            Sum.append(Out)
            j = j + 1
        ListOfMSE.append(np.sum(Sum))
        index = index + 1  # Use index instead of i

    MinK = np.argmin(ListOfMSE)
    ListofK.append(MinK)


    i = i + 1


k_List=ListofK  
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




i=1
X_hat_List=[]
while i<11:
    for j in [0,1,2]:
        if ListofParam[i][j]==0:
            ListofParam[i][j]=0.01
    pVec=np.linspace(0.001, 0.1, 100)
    DataList=np.array(df.iloc[:,i])
    First=[1 for x in DataList if x>=k_List[i]]
    x_hat=k_List[i]+(ListofParam[i][2]/ListofParam[i][0])*(((np.sum(First)/(len(DataList)*pVec))**(ListofParam[i][0]))-1)
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

