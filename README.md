# PassPredictModel
Model that predicts whether a student will pass the exams

## Requirements
- Python 3.7
- Pyspark
- Numpy

## Usage
1 Download **StudentsPerformance.csv**
2 Functions that convert CSV input data into numerics
    
        def binary(YN):
            if (YN == 'Y'):
                return 1
            else:
                return 0
