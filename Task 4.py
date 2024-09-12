#Task 4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

"""
df = pd.read_csv('C:/Users/hp/Desktop/Intership/Prodigy Task 4/twitter_training.csv', header=None)

#Naming the header
new_headers = ['Id', 'Topic', 'Sentiment','Review']
df.columns = new_headers
df.to_csv('dataset.csv', index=False)
"""


df = pd.read_csv('C:/Users/hp/Desktop/Intership/Prodigy Task 4/dataset.csv')

print(df.head())
print(df.tail())

print(df.info())
print(df.describe())


#Removing duplicates
print("duplicates:",df.duplicated().sum())
a=df.drop_duplicates(inplace=True)
print("duplicates after removing them:",a)

#Missing values
print("Missing values:",df.isna().sum())
b=df.dropna(inplace= True)
print("Missing values after removing them:",b)

print(df.shape)


#Visualization

#Sentiment Distribution
plt.hist(df['Sentiment'])
plt.title("Sentiment Distribution")
plt.show()

#Pie Chart sentiment
sentiment_counts = df['Sentiment'].value_counts()
plt.pie(sentiment_counts.values, labels=sentiment_counts.index,autopct='%1.1f%%',colors=['#66b3ff','#99ff99','#ff9999','#ffcc99'])
plt.title("Pie Chart about Sentiment Distribution")
plt.show()

#Topic Distribution
trend_counts = df['Topic'].value_counts()
plt.figure(figsize=(15,10))
sns.barplot(x=trend_counts.values, y=trend_counts.index)

plt.xlabel('Count')
plt.ylabel('Topic')
plt.title('Bar Plot of Topic Column')
plt.show()

#Stacked Bar
trend_sentiment = pd.crosstab(df['Topic'], df['Sentiment'])
trend_sentiment.plot(kind='bar', stacked=True, figsize=(10,7), color=['#66b3ff','#99ff99','#ff9999'])

plt.xlabel('Topic')
plt.ylabel('Count')
plt.title('Stacked Bar Chart of Trend and Sentiments')
plt.show()


#Positive, negative , neutral and irrelevant sentiment
# Get the top 10 most frequent trends
top_10_trends = df['Topic'].value_counts().nlargest(10).index

# Filter the dataset for only the top 10 trends
top_10_data = df[df['Topic'].isin(top_10_trends)]

fig, axes = plt.subplots(4, 1, figsize=(34, 24))
fig.suptitle('Sentiment Distribution Across Top 10 Trends', fontsize=18)
sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']

for sentiment, ax in zip(sentiments, axes.flatten()):
    filtered_data = top_10_data[top_10_data['Sentiment'] == sentiment]
    sns.countplot(data=filtered_data, x='Topic', ax=ax, palette='viridis', order=top_10_trends)
    
    ax.set_title(f'{sentiment} Sentiment')
    ax.set_xlabel('Topic')
    ax.set_ylabel('Count')

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()


#WordCloud

sentiment_list = ["Positive", "Neutral", "Negative", "Irrelevant"]
colormap_list = ["YlGn_r", "Blues_r", "Reds_r", "copper_r"]
ax_list = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Create a set of stopwords
stopwords_set = set(STOPWORDS)

# Creating subplots for the word clouds
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))

for sentiment, (row, col), colormap in zip(sentiment_list, ax_list, colormap_list):
    # Filter the text based on the sentiment
    text = " ".join(review for review in df[df["Sentiment"] == sentiment]["Review"].astype(str))
    
    # Create a word cloud for the specific sentiment
    wordcloud = WordCloud(colormap=colormap, stopwords=stopwords_set, width=1600, height=900).generate(text)
    
    # Plot the word cloud
    ax[row, col].imshow(wordcloud, interpolation='bilinear')
    ax[row, col].set_title(sentiment + " WordCloud", fontsize=18)
    ax[row, col].axis('off')

# Adjust layout
fig.tight_layout()
plt.show()

#Decision tree
label_encoder = LabelEncoder()

df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])
df['Topic'] = label_encoder.fit_transform(df['Topic'])

# Define the feature variables (X) and the target variable (y)
X = df[['Topic']]  # Features (you can add more features if needed)
y = df['Sentiment']  # Target (Sentiment)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the classifier
decision_tree.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = decision_tree.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot the Decision Tree
plt.figure(figsize=(12, 8))  # Set the figure size for better clarity
plot_tree(decision_tree, feature_names=['Topic'], class_names=label_encoder.classes_, filled=True, rounded=True, fontsize=10)

plt.title('Decision Tree for Sentiment Classification')
plt.show()
