# PassPredictModel
Model that predicts whether a student will pass the exams

## Requirements
- Python 3.7
- Pyspark
- Numpy

## Usage
1 Download **StudentsPerformance.csv**

2 Functions that convert CSV input data into numerics
ex) binary function to return 1 for Yes and 0 for No
    
    def binary(YN):
        if (YN == 'Y'):
            return 1
        else:
            return 0


3 Convert data in CSV to a Labeled Point that MLLib can use

    def createLabeledPoints(fields):
        gender = binary(fields[0])
        race = mapRace(fields[1])
        parent = mapLevel(fields[2])
        lunch = binary(fields[3])
        testprepare = binary(fields[4])
        math = int(fields[5])
        reading = int(fields[6])
        writing = int(fields[7])
        passed = mapScore(fields[9])

        return LabeledPoint(passed, array([gender, race,
            parent, lunch, testprepare, math, reading, writing]))
   
 
4 Load up CSV file

    rawData = sc.textFile("StudentsPerformance.csv")
    header = rawData.first()
    rawData = rawData.filter(lambda x:x != header)
    

5 Train the data

    trainingData = csvData.map(createLabeledPoints)

6 Create Test Candidate
Put in values to test

    testCandidates = [ array([1, 2, 5, 1, 1, 70, 80, 74])]
    testData = sc.parallelize(testCandidates)
    
7 Train decision tree classifier

    model = DecisionTree.trainClassifier(trainingData, numClasses=2,
                                         categoricalFeaturesInfo={0:2, 1:6, 2:7, 3:2, 4:2},
                                         impurity='gini', maxDepth=5, maxBins=32)
   

## Result
If average score is 1, then pass

    predictions = model.predict(testData)
    print('Pass Prediction:')


    results = predictions.collect()
    for result in results:
        if result == 0:
            print("Non-pass")
        elif result == 1:
            print("Pass!")
