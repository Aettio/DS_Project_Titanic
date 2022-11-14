# Data science | Project : "Titanic"

## Project guide

Before the beginning:
- "README.txt" - Contains the project itself with visualizations. It is assumed that the reader will view it as the main input file.
- "DataFrames" - Contains all datasets.
- "Images" - Contains all images "README.txt".
- "Code_Visual" - Contains the code for the entire visual (Charts and 3D).
- "Code_Tree" - Contains the tree algorithm itself.

p.s. Additional notes have been made throughout the code for ease of reading and understanding.

## Sections

- Introduction
- A task
- Exploratory Data Analysis (Data analysis, more details can be viewed in Code_Visuals.py)
   - Data cleaning
   - Search for correlations
   - Visualization
   - Choice of parameters for learning
- Classification Tree (Distribution algorithm, more details can be viewed in Code_Tree.py)
   - Cleaning and preparing data for training
   - Primary training
   - Selection of optimal tree parameters
   - Testing
- Total
- Sources

## Introduction

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely known "unsinkable" Titanic sank after colliding with an iceberg. Unfortunately, there were not enough lifeboats on board for everyone, resulting in the deaths of 1,502 out of 2,224 passengers and crew.

Although there was a certain element of luck in survival, it seems that certain groups of people were more likely to survive than others.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Titanic_picture.jpg)

## A task

In this task, we will build a predictive model that answers the question, “Which types of people are more likely to survive?” using passenger data (for example, name, age, gender, socioeconomic class, etc.). And also analyze what factors influenced survival and to what extent.

For this, 3 datasets are provided (which you can view in the "DataFrames" folder):
- "gender_submission.csv"
- "train.csv"
- "test.csv"

p.s. While working on the project, I additionally created new "clean" sets (Without everything that was superfluous for me and which I did not upload to Git). These sets were made for my personal convenience, so you can see them in some parts of the code.

## EDA (Exploratory Data Analysis)

After cleaning the data, it was necessary to highlight the main dependencies and conduct a small analysis of the relationships. You also need to be sure that they make sense. To start the analysis, I chose a graph to visualize correlations.

### Correlations

On this graph, it is immediately worth noting several correlations that are interesting to us.
- "Sex" (Gender)
- "Pclass" (Class)
- "Fare" (Fee)
- "Age" (Age)
Ignoring "SibSp" and "Parch" (this is a logical correlation of members of the same family), in this case we are most interested in the "Survived" column.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Корреляции.png)


### Visualization of survivors by gender

Here we test the first major relationship. In this graph, we see that the probability of survival is approximately 2 times higher if the gender is female.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Выжившие_по_полу.png)


### Percentage of survivors by class

In this graph, we have taken the percentage of survival to compensate for the difference in the number of people in different classes. Here we note that the 1st class had an extremely high probability of survival, compared to the 2nd and 3rd. Presumably this could be due to the higher location of the cabins.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Выживших_по_классу.png)


### Visualization of survivors by age

The analysis of the correlation of age caused ambiguous thoughts in me, so it was decided to go deeper with the help of 3D visualization.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Выжившие_по_возрасту.png)


### General 3D visualization

Having considered this visualization, I assumed that although age influenced survival, since children mostly survived, it was also necessary to take into account that there were practically no old people on the ship. But I still had a hunch that there was a class relationship worth checking. So I continued.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Visual_3D.png)

### 3D visualization "Fare"

It was necessary to check this correlation, since it raised the most questions for me. Subsequently, I still discovered it, although I had to play with different charts, the reason for this was some outliers, which in turn made the standard charts little understandable.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Visual_3D_Fare.png)

(In this chart, the most noticeable shift is upward along the "Fare" line, or in this case it is the z-axis)

### Survivors by class versus age

And then I made graphs by age for each class, as well as their survival rate. As a result, as one would expect, the general trend showed a shift to the right side in age (that is, the higher the class, the older). Which had such an effect on the correlation of age to survival, which is essentially logical. Older people generally have more wealth, which eventually allowed them to take a higher class. As a result, you need to be more careful with age when setting up the algorithm.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Выжившие_по_классу_в_соотношении_с_возрастом.png)

(For the compilation of all graphs by age, columns in which the NA value in the "Age" column were not used)

## Classification Tree

I had 2 choices, create a single tree or forest. The forest would presumably give a more accurate classification and would probably be a better solution. But in this case, the goal of my personal interest was to create a single classification tree and make it as accurate as possible.

### Cleaning and preparing data for training

Before starting the task, we were given 3 data sets (Can be viewed in the "DataFrames" folder):
- "gender_submission.csv" - Set with information on Id and survival.
- "train.csv" - Set on which we will train and test our algorithm.
- "test.csv" - The purpose of our task (Predict this data).

We immediately connect "gender_submission.csv" with "train.csv" by the key "PassengerId" and remove all unnecessary columns except for those that we noted for ourselves in the EDA (They have a correlation with survival).

### Initial training

After the initial training, as expected, we got a retrained tree. Next, we begin to adjust the parameters and cut the possibilities of the tree in order not to give it a chance to retrain.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/First_Tree.png)

### Selection of optimal tree parameters

For this step, we create a graph with visualization of different parameters for the tree.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Tree_Accuracy.png)

Now we understand at what stage the retraining of our tree begins and we can correct it. Based on this graph, we adjust the optimal depth parameters, as well as the minimum number of samples for splitting the branch, and test. We get an accuracy result of ≈ 0.83 (Rounded according to the rules), respectively, 83%.

### Testing

We conduct Cross Value Score out of 5. For more statistics and understanding of the trend in order to avoid errors.

We get: 0.77094972, 0.78089888, 0.84831461, 0.79213483, 0.81460674.

Average accuracy: 0.8014 with a standard deviation of +/- 0.0552.

Based on a decision tree classification model, I predicted whether the passengers would survive with a ≈80% chance.

## Total

I added a couple more lines to visualize the tree itself, so that everything is clearer and this is what happened. The prediction percentage is not bad, given that this tree can be used to analyze any randomly taken data from the titanic. I tried to find the optimal balance between its sizes so that the tree would not be overtrained. As a result, it turned out ≈80% or 4 out of 5 the algorithm classifies correctly. It is possible to achieve higher classification values, but then the algorithm will be less universal.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Final_Tree.png)

(You can view the full code in Code_Tree.py)

## Sources

- Dataset : https://www.kaggle.com/c/titanic
- Matplotlib documentation : https://matplotlib.org/3.1.1/contents.html
- Seaborn documentation : https://seaborn.pydata.org/introduction.html
- Pandas documentation : https://pandas.pydata.org/pandas-docs/stable/reference/frame.html
- Plotly 3D documentation : https://plotly.com/python/3d-scatter-plots/
