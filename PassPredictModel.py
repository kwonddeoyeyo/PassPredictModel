from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf = conf)

# Some functions that convert our CSV input data into numerical
# features for each job candidate
def binary(YN):
    if (YN == 'Y'):
        return 1
    else:
        return 0
    
def mapRace(group):
    if (group == 'A'):
        return 1
    elif (group == 'B'):
        return 2
    elif (group == 'C'):
        return 3
    elif (group == 'D'):
        return 4
    elif (group == 'E'):
        return 5
    else:
        return 0

def mapLevel(degree):
    if (degree == 'SHS'):
        return 1
    elif (degree == 'HS'):
        return 2
    elif (degree == 'SC'):
        return 3
    elif (degree == 'BD'):
        return 4
    elif (degree == 'AD'):
        return 5
    elif (degree == 'MD'):
        return 6
    else:
        return 0
    
def mapScore(score):
    if int(score) >= 70:
        return 1
    else:
        return 0

# Convert a list of raw fields from our CSV file to a
# LabeledPoint that MLLib can use. All data must be numerical...
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

#Load up our CSV file, and filter out the header line with the column names
rawData = sc.textFile("StudentsPerformance.csv")
header = rawData.first()
rawData = rawData.filter(lambda x:x != header)

# Split each line into a list based on the comma delimiters
csvData = rawData.map(lambda x: x.split(","))

# Convert these lists to LabeledPoints
trainingData = csvData.map(createLabeledPoints)

# Create a test candidate
testCandidates = [ array([1, 2, 5, 1, 1, 70, 80, 74])]
testData = sc.parallelize(testCandidates)

# Train our DecisionTree classifier using our data set
model = DecisionTree.trainClassifier(trainingData, numClasses=2,
                                     categoricalFeaturesInfo={0:2, 1:6, 2:7, 3:2, 4:2},
                                     impurity='gini', maxDepth=5, maxBins=32)

# Now get predictions for our unknown candidates. (Note, you could separate
# the source data into a training set and a test set while tuning
# parameters and measure accuracy as you go!)
predictions = model.predict(testData)
print('Pass Prediction:')
results = predictions.collect()
for result in results:
    if result == 0:
        print("Non-pass")
    elif result == 1:
        print("Pass!")

# We can also print out the decision tree itself:
print('Learned classification tree model:')
print(model.toDebugString())
# Changes
