# ****News sentiment classifier using Support Vector Machine****

### ***Data collection:***
The data used for this project was collected from the same source, however the initial model training was made on a static dataset from Kaggle.  (Saksham Kumar 2024)The same Kaggle user provides the script utilized for retrieving data from the NewsAPI.  (NewsAPI 2024)
This script has been utilized in a modified version, to collect the new data for the project. The script collect new data every 24 hours, which is then stored a SQL database located on the virtual machine that is also running the python environment used in this project.
***Data preprocessing:***
The raw article data retrieved from the API is transformed into a structured dataframe. Here articles marked as '[Removed]' are filtered out and relevant column are selected: source, title, description, url, urltoImage, publishedat and content. Publication date is also standardized. 
The text data was converted into numerical features using TF-IDF vectorization. 

### ***Model selection***
Multiple models were considered for the purpose of creating a news sentiment analyser. The choice fell on Support Vector Machines (SVMs) as they are well-suited for news sentiment analysis due to their effectiveness in high-dimensional spaces and memory efficiency, especially considering the large dataset used to train it. SVMs work by trying to find the hyperplane, which best seperates the text samples into different sentiment categories. (scikit-learn 2024) 
Additionaly the implementation of the Support Vector Machine was easy, and since the project was mainly focused on making the pipeline work from backend to frontend, the model choice and performance was not deemed a priority.

Deep learning and the implementation of large language models was also in consideration for the purpose of improving the sentiment labelling, as some of the sentiment results of the SVM model were blatantly wrong. Using Together.AI both the pretrained models togethercomputer/RedPajama-INCITE-7B-Base and meta-llama/Meta-Llama-3-70B were used with different types of prompting. In the end, in part due to time constraint, we did not manage to make it work and decided to go with the SVM for our model choice.
 
### ***Hyperparameter tuning & Model Evaluation***
In an attempt to create the best performing SVM model, the model was trained on both balanced and unbalanced data. Class weight were utilized to attempt to address class imbalance. N-grams was used for attempting different combinations of tokens. Baggingclassifier was also used in an attempt to make the model better.
To find the optimal parameters for the model both Random Search and Grid Search were implemented, to find the best parameter tuning. Specifically the target was for which parameters would give the highest accuracy. The best results we were able to achieve with the model was with ‘C’ parameter at 10 and with the kernel set to the Radial Basis Function:
Evaluation Metric	Accuracy	Precision	Recall	F1 score
Score	0.845	0.8392	0.845	0.8362

### ***Model deployment***
The model was deployed using Streamlit, which serves as the frontend of the application. The model itself, as well as the SQL database, which contains the news articles are all hosted on ucloud on a virtual machine. 
The database received new data from the data collection script every 24 hours, and the Streamlit updates to the newest data available upon every reload, which means it should have the same data for 24 hours before updating. 
The streamlit displays model evaluations metrics and gives the option to choose between different news outlet, which will display upto the 10 newest articles from the selected outlet and their sentiment, it also includes a piechart, which shows the sentiment distribution for all of the news included from that news source. The Streamlit app also has an option to predict sentiment from user input in the form of headlines.

### ***Visualisation of pipeline***
 
![image](https://github.com/jogfx/MLops-exam/assets/71497575/facab2f6-cc98-4dfe-8fca-ce2c6e5b7498)

 
### ***Bibliography:***
NEWSAPI, 2024-last update, Search worldwide news with code . Available: https://newsapi.org/ [06/05/, 2024].
SAKSHAM KUMAR, January, 2024-last update, Global News Dataset . Available: https://www.kaggle.com/datasets/everydaycodings/global-news-dataset [06/05/, 2024].
SCIKIT-LEARN, 2024-last update, Support Vector Machines. Available: https://scikit-learn.org/stable/modules/svm.html [06/05/, 2024].

