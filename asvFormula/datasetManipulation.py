from sklearn.preprocessing import LabelEncoder

def encodeCategoricalColumns(dataset):
    encodingDict = {}
    le = LabelEncoder()
    encodedDataset = dataset.copy()
    categorical_columns = dataset.select_dtypes(include=['object', 'category', 'bool']).columns
    for columnName in categorical_columns:
        encodedDataset[columnName] = le.fit_transform(encodedDataset[columnName])
        encodingDict[columnName] =  le.classes_
    return encodingDict, encodedDataset