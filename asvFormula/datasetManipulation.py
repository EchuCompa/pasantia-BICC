from sklearn.preprocessing import LabelEncoder

def encodeCategoricalColumns(dataset):
    encodingDict = {}
    le = LabelEncoder()
    encodedDataset = dataset.copy()
    categorical_columns = dataset.select_dtypes(include=['object', 'category', 'bool']).columns
    for column in categorical_columns:
        encodedDataset[column] = le.fit_transform(encodedDataset[column])
        encodingDict[column] =  dict(zip(le.classes_, range(len(le.classes_))))
    return encodingDict, encodedDataset