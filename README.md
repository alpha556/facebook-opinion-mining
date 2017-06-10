# facebook-opinion-mining
Day by day the number of text documents in digital form is increasing. Text classification is used to organize these text documents. Our project uses Text Classification on Social Media  (Facebook) for mining user opinion. Since text classification has the problem of high dimensionality of feature space, feature selection and feature extraction methods are used to improve the performance of text categorization.
The proposed methodology comprises of data pre-processing followed by training a supervised machine learning classifier using the preprocessed data. The data pre-processing step involves removal of stop words and stemming. Feature extraction is done using a bag of words approach with Term Frequency and Inverse Document Frequency Weights. The feature vector to be provided for data mining is then formulated through the identified root words. Feature selection procedure is adopted to attain the most significant features. Various classification algorithms were implemented on the reduced feature set to classify the emotions as positive and negative. Support Vector Machine provided the best possible results achieving an accuracy of approx 80%.The trained model is then applied on Facebook Posts and Comments for Opinion Mining.
