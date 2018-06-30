
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <stdint.h>
#include <string.h>

// TODO: Need these values to be more dynamic. I want them dependent on feature values or something in training set
#define MAX_DATA_RANGE 75
#define MIN_DATA_RANGE 50

//#define LEARNING_RATE .00000005
#define LEARNING_RATE 0.000000135
//#define LEARNING_RATE .0000001
//#define LEARNING_RATE .3 // For 1 feature training data

#define TRAINING_DATA_FILE "training_data.txt"

float EULERS_NUMBER;

typedef struct TrainingSet
{
    int size;
    int numFeatures;
    int numTargets;
    float** features;
    float* targets;
} TrainingSet;

typedef struct CostFunction
{
    int numParameters;
    float* parameters;
} CostFunction;

float Exponentiate(float base, int power);
float Logarithm(float argument, float base);
float SquareRoot(float radicand);
float AbsoluteValue(float val);
float LinearRegressionHypothesis(float* features, float* thetas, int numParameters);
float LogisticRegressionHypothesis(float* features, float* thetas, int numParameters);
void PrintTrainingSet(TrainingSet* trainingSet);
float StandardDeviation(float* featureValues, float mean, int sampleSize);
void GrabVector(float** matrix, int numRows, int numCols, char* direction, int desiredIndex, float* out);
float CalculateAverageOfArray(float* arrayOfValues, int size);
void ApplyFeatureScaling(TrainingSet* trainingSet, uint8_t applyMeanNormalization);
void ScaleTargets(TrainingSet* trainingSet);

uint8_t ValidateObject(void* ptr, char* objectName)
{
    if(!ptr)
    {
        printf("ERROR: Failed to allocate memory for %s!\n", objectName);
        exit(1);
    }
}

void ReadTrainingData(TrainingSet* trainingSet)
{
    if( !trainingSet || !(trainingSet->targets) || !(trainingSet->features) )
    {
        printf("Training set is not allocated!\n");
        exit(1);
    }

    FILE* fp = fopen(TRAINING_DATA_FILE, "rw");

    if(!fp)
    {
        printf("Training data file did not open successfully!\n");
        exit(1);
    }

    char ch;
    uint8_t onFeature = 1;
    uint8_t decimalPoint = 0;
    int numDecimalPlaces = 0;
    int numTrainingSamplesRead = 0;
    int numFeatures = 0;
    int data = 0;

    while( (ch = fgetc(fp)) != EOF && ch != '-' )
    {
        // Just eat up everything before the "-------" section of the file (used for notes and such)
    }

    // Allocate space for 2 features, since all feature vectors will have a first feature component that will always be 0
    // This is because we want the equation -> theta0 * x0^0 + theta1 * x1^1 + ... + thetan * xn^n for n features
    // We want this equation to later become theta0 + theta1*x1^1... So we need some value for x0
    trainingSet->features[trainingSet->size] = malloc(sizeof(float) * 2);
    trainingSet->features[trainingSet->size][numFeatures++] = 1; // Fill the first feature with a 1, as mentioned above

    while( (ch = fgetc(fp)) != EOF )
    {
        if(feof(fp))
        {
            printf("YOOO");
            break;
        }

        if( onFeature )
        {
            //printf("FEATURE\n");
            if( isdigit(ch) )
            {
                if(decimalPoint)
                    numDecimalPlaces++;

                data = (data * 10) + (ch - '0');
            }
            else if(ch == ',' || ch == ')')
            {
                trainingSet->features[trainingSet->size] = realloc(trainingSet->features[trainingSet->size], 
                                                                    sizeof(float) * (numFeatures + 1));
                trainingSet->features[trainingSet->size][numFeatures++] = data;

                if(decimalPoint)
                    trainingSet->features[trainingSet->size][numFeatures-1] /= (numDecimalPlaces * 10);

                data = 0;
                decimalPoint = 0;
                numDecimalPlaces = 0;
            }
            else if( ch == '\n' && numFeatures > 1 )
            {
                // This assumes that we will have the same number of features for each training example, which is a fair assumption
                trainingSet->numFeatures = numFeatures;
                onFeature = 0;
                numFeatures = 0;
            }
            else if(ch == '.')
            {
                decimalPoint = 1;
            }
        }
        else
        {
            //printf("HI\n");
            if( isdigit(ch) )
            {
                if(decimalPoint)
                    numDecimalPlaces++;

                data = (data * 10) + (ch - '0');
            }
            else if(ch == ',' || ch == ')')
            {
                trainingSet->targets = realloc(trainingSet->targets, sizeof(float) * (trainingSet->numTargets + 1));
                trainingSet->targets[trainingSet->numTargets++] = data;

                if(decimalPoint)
                    trainingSet->targets[trainingSet->numTargets-1] /= (numDecimalPlaces * 10);

                data = 0;
                decimalPoint = 0;
                numDecimalPlaces = 0;
            }
            else if( ch == '\n' )
            {
                //trainingSet->size++;
                trainingSet->features = realloc(trainingSet->features, sizeof(float*) * (++trainingSet->size + 1));
                trainingSet->features[trainingSet->size] = malloc(sizeof(float) * 2);
                trainingSet->features[trainingSet->size][numFeatures++] = 1; // Fill the first feature with a 1, as mentioned above
                onFeature = 1;
            }
            else if(ch == '.')
            {
                decimalPoint = 1;
            }
        }
    }

    printf("Training set size: %i\n", trainingSet->size); // Compensate for 0-based size above


    fclose(fp);
}

// TODO: Probably could do away with the pointers for max and min feature values (since all is done within the first outer for loop
void ApplyFeatureScaling(TrainingSet* trainingSet, uint8_t applyMeanNormalization)
{
    float* maxFeatureVal = malloc(sizeof(float) * trainingSet->numFeatures);
    float* minFeatureVal = malloc(sizeof(float) * trainingSet->numFeatures);

    float* featureValRange = malloc(sizeof(float) * trainingSet->numFeatures);

    float* featureValSum = malloc(sizeof(float) * trainingSet->numFeatures);

    float* standardDeviation = malloc(sizeof(float) * trainingSet->size);

    for(int featureIndex = 0; featureIndex < trainingSet->numFeatures; featureIndex++)
    {
        featureValSum[featureIndex] = 0;

        maxFeatureVal[featureIndex] = trainingSet->features[0][featureIndex];
        // If only feature, assume range is from 0 to max value... TODO: Does this have any nasty side effects?
        minFeatureVal[featureIndex] = 0; // Don't want anything subtracted from max if this is the only feature

        for(int trainingExampleIndex = 0; trainingExampleIndex  < trainingSet->size; trainingExampleIndex ++)
        {
            if( trainingSet->features[trainingExampleIndex][featureIndex] > maxFeatureVal[featureIndex] )
                maxFeatureVal[featureIndex] = trainingSet->features[trainingExampleIndex][featureIndex];
            else if( trainingSet->features[trainingExampleIndex][featureIndex] < minFeatureVal[featureIndex] )
                minFeatureVal[featureIndex] = trainingSet->features[trainingExampleIndex][featureIndex];

            //printf("Feature value at [%i][%i]: %f\n", trainingExampleIndex, featureIndex, trainingSet->features[trainingExampleIndex][featureIndex]);
            featureValSum[featureIndex] += trainingSet->features[trainingExampleIndex][featureIndex];
        }

        //printf("Max feature value: %f\tMin feature value: %f\n", maxFeatureVal[featureIndex], minFeatureVal[featureIndex]);
        featureValRange[featureIndex] = maxFeatureVal[featureIndex] - minFeatureVal[featureIndex];
        //printf("featureValRange[%i] = %f\n", featureIndex, featureValRange[featureIndex]);
    }

    float* averageFeatureVal = calloc(trainingSet->numFeatures, sizeof(float));
    for(int i = 0; i < trainingSet->numFeatures; i++)
    {
        averageFeatureVal[i] = featureValSum[i] / trainingSet->size;

        int numRows = trainingSet->size;
        int numCols = trainingSet->numFeatures;
        float* featureVector = malloc(sizeof(float) * numRows);
        GrabVector(trainingSet->features, numRows, numCols, "column", i, featureVector); // Grab the column representing the feature vector
        standardDeviation[i] = StandardDeviation(featureVector, averageFeatureVal[i], trainingSet->numFeatures);
        free(featureVector);

        if(standardDeviation[i] == 0)
            standardDeviation[i] = 1;

        //printf("standardDeviation[%i] = %f\n", i, standardDeviation[i]);
        //printf("Feature Value Sum[%i]: %f\n", i, featureValSum[i]);
        //printf("TrainingSetSIze = %i\n", trainingSet->size);
        //printf("averageFeatureVal = %f\n", averageFeatureVal[i]);
        //getchar();
    
        //printf("Feature val sum: %f\n", featureValSum[i]);
        //printf("Average feature val: %f\n", averageFeatureVal[i]);
    }

    for( int i = 0; i < trainingSet->size; i++ )
    {
        for( int j = 0; j < trainingSet->numFeatures; j++ )
        {
            //printf("Feature value range = %f\n", featureValRange[j]);
            //printf("Feature %i before: %f\n", j, trainingSet->features[i][j]);
            // Scale the feature down between a range (0 - 1)
            //trainingSet->features[i][j] /= featureValRange[i]; // This is using the max-mix method
            trainingSet->features[i][j] /= standardDeviation[j]; // This is like above, but using the stnandard deviation method
            //printf("standard deviation[%i]: %f\n", j, standardDeviation[j]);
            //printf("Feature after dividing out standardDeviation: %f\n", trainingSet->features[i][j]);

            if(applyMeanNormalization)
            {
                //trainingSet->features[i][j] -= (averageFeatureVal[j] / featureValRange[i]); // Using the max-mix method
                trainingSet->features[i][j] -= (averageFeatureVal[j] / standardDeviation[j]); // Using standard deviation method
                //printf("Average Feature Value: %f\n", averageFeatureVal[j]);
                //printf("Subtrahend: %f\n", averageFeatureVal[j] / standardDeviation[j]);
            }
            else // This is in reference to wikipedia's version of feature scaling by means of "rescaling"
            {
                trainingSet->features[i][j]; // -= minFeatureVal[j]; Assuming I don't want this, since negatives? But wikipedia?
            }

            //printf("Feature %i after: %f\n", j, trainingSet->features[i][j]);
        }
    }

    free(maxFeatureVal);
    free(minFeatureVal);
    free(featureValRange);
    free(featureValSum);
    free(standardDeviation);
    free(averageFeatureVal);
}

// TODO: This does not work nicely (: I may want to get a larget scale than 0 to 1 for features who go into the 100s of thousands (:
void ScaleTargets(TrainingSet* trainingSet)
{
    float averageTarget = CalculateAverageOfArray(trainingSet->targets, trainingSet->size);
    printf("Average Target Value = %f\n", averageTarget);

    for(int i = 0; i < trainingSet->size; i++)
    {
        trainingSet->targets[i] /= averageTarget;
        printf("%i: Target old value = %f\t Target new value %f\n", i, trainingSet->targets[i] * averageTarget, trainingSet->targets[i]);
    }
}

TrainingSet* CreateTrainingSet()
{
    TrainingSet* trainingSet = malloc(sizeof(TrainingSet));

    trainingSet->size = 0;
    trainingSet->numFeatures = 0;
    trainingSet->numTargets = 0;

    trainingSet->features = malloc(sizeof(float*) * 1);
    trainingSet->targets = malloc(sizeof(float) * 1);

    ReadTrainingData(trainingSet);

    return trainingSet;
}

void DestroyTrainingSet(TrainingSet* trainingSet)
{
    for(int i = 0; i < trainingSet->numFeatures; i++)
    {
        free(trainingSet->features[i]);
    }

    free(trainingSet->features);
    free(trainingSet->targets);
    free(trainingSet);
}

void PrintTrainingSet(TrainingSet* trainingSet)
{
    printf("\n");
    for(int i = 0; i < trainingSet->size; i++)
    {
        printf("Training Set #%i\n", i);
        printf("----------------\n");

        printf("Features: ");
        for(int j = 0; j < trainingSet->numFeatures; j++)
        {
            printf("%0.2f", trainingSet->features[i][j]);
            
            if(j != trainingSet->numFeatures - 1)
                printf(", ");
            else
                printf("\n");
        }
        printf("Target: %0.2f\n\n", trainingSet->targets[i]);
    }

    printf("\n");
}

CostFunction* CreateCostFunction(int numFeatures)
{
    CostFunction* costFunction = malloc(sizeof(CostFunction));
    
    costFunction->numParameters = numFeatures;
    costFunction->parameters = malloc(sizeof(float) * costFunction->numParameters);

    // Some random start point for our thetas
    for(int i = 0; i < costFunction->numParameters; i++)
    {
        // TODO: Make this more dynamic and scalable for MAX_DATA_RANGE and MIN_... maybe try depending on max/min of data?
        costFunction->parameters[i] = ( (rand() % (MAX_DATA_RANGE - MIN_DATA_RANGE)) + MIN_DATA_RANGE );
    }

    return costFunction;
}

void DestroyCostFunction(CostFunction* costFunction)
{
    free(costFunction->parameters);
    free(costFunction);
}

float RunLinearRegressionCostFunction(CostFunction* costFunction, TrainingSet* trainingSet)
{
    float cost_function_sum = 0;

    // Find the sum that is used for each of the parameters
    for(int i = 0; i < trainingSet->size; i++)
    {
        float estimated_value = LinearRegressionHypothesis(trainingSet->features[i], costFunction->parameters, costFunction->numParameters);
        cost_function_sum += Exponentiate( (estimated_value - trainingSet->targets[i]), 2 );
    }

    return ( cost_function_sum / (float) (2.0 * trainingSet->size) );
}

float RunLogisticRegressionCostFunction(CostFunction* costFunction, TrainingSet* trainingSet)
{
    float cost_function_sum = 0;

    for(int i = 0; i < trainingSet->size; i++)
    {
        cost_function_sum += trainingSet->targets[i] * Logarithm(
            LogisticRegressionHypothesis(trainingSet->features[i], costFunction->parameters, costFunction->numParameters), 
            10
        );

        cost_function_sum += (1 - trainingSet->targets[i]) * Logarithm(
            1 - LogisticRegressionHypothesis(trainingSet->features[i], costFunction->parameters, costFunction->numParameters),
            10
        );
    }
}

// Predict target (y value) given a feature (x value)
// Using "theta_0 + theta_1*x" as hypothesis function
float LinearRegressionHypothesis(float* features, float* thetas, int numParameters)
{
    float hypothesisSum = 0;
    for(int i = 0; i < numParameters; i++)
    {
        hypothesisSum += thetas[i] * Exponentiate(features[i], i);

        //printf("thetas[%i] = %f\tfeatures[%i] = %f\n", i, thetas[i], i, features[i]);
    }
    
    //printf("hyptohesisSum = %f\n", hypothesisSum);

    return hypothesisSum;
}

// The hypothesis for Logistic Regression. It uses the Logistic (Sigmoid) function.
float LogisticRegressionHypothesis(float* features, float* thetas, int numParameters)
{
    // Linear Regression's hypothesis finds Theta(transpose) * x for me (Transpose of Parameter vector multiplied by Feature vector)
    float argument = LinearRegressionHypothesis(features, thetas, numParameters);
    return (1 / (1 + Exponentiate(EULERS_NUMBER, -argument)));
}

// TODO TODO TODO: Ensure that this is correct. Check for an actually good method online... 
float Logarithm(float argument, float base)
{
    /*float precision = .001;

    float a = (1 + argument) / 2.0;
    float b = SquareRoot(argument);

    while( AbsoluteValue(a - b) >= precision )
    {
        printf("A= %f\tB= %f\n", a,b);
        a = (a + b) / 2.0;
        b = SquareRoot(a * b);
        printf("RESULT: %f\n", 2 * ( (argument - 1) / (a + b) ));
    }

    float result = 2 * ( (argument - 1) / (a + b) );
    
    printf("Logarithm, base %f, of %f = %f\n", base, argument, result);
    return result;*/



    float new = argument / base;
    if((int) new == 1)
        return 1;
    
    return 1 + Logarithm(new, base);
}

float Exponentiate(float base, int power)
{
    if(power < 0)
    {
        base = (1/base);
    }

    float result = 1;
    for(int i = 0; i < power; i++)
    {
        result *= base;
    }

    //printf("%f to the power of %i = %f\n", base, power, result);

    return result;
}

float CalculateAverageOfArray(float* arrayOfValues, int size)
{
    float sumAcc = 0;
    for(int i = 0; i < size; i++)
    {
        sumAcc += arrayOfValues[i];
    }

    return sumAcc / size;
}

float SquareRoot(float radicand)
{
    if( !radicand )
        return 0;
    else if( radicand < 0 )
    {
        printf("Result is imaginary!\n");
        return -1;
    }

    float result = 0;
    float precision = 0.01;
    int numIterations = 0;

    int leftHandSquare;
    int rightHandSquare;

    for(int i = 0; i < radicand; i++)
    {
        if( i * i < radicand )
        {
            leftHandSquare = i;
            rightHandSquare = i;
        }
        else if( i * i > radicand )
        {
            rightHandSquare = i;
            break;
        }
        else
        {
            result = i;
            break;
        }
        numIterations++;
    }

    float lhsApprox = leftHandSquare;
    float rhsApprox = rightHandSquare;
    //printf("lhsApprox = %f\n", lhsApprox);
    //printf("rhsApprox = %f\n", rhsApprox);
    //printf("Result: %i\n", result);

    while(!result)
    {
        lhsApprox += precision;
        rhsApprox -= precision;
        //printf("Adjusted lhsApprox = %f\n", lhsApprox);
        //printf("Adjusted rhsApprox = %f\n", rhsApprox);

        if(lhsApprox * lhsApprox >= radicand)
        {
            //printf("APPROX FROM LHS: %f\n", lhsApprox);
            result = lhsApprox;
            break;
        }
        else if( leftHandSquare != rightHandSquare && rhsApprox * rhsApprox <= radicand)
        {
            //printf("APPROX FROM RHS: %f\n", rhsApprox);
            result = rhsApprox;
            break;
        }
        numIterations++;
    }

    //printf("leftHandSquare = %i\n", leftHandSquare);
    //printf("rightHandSquare = %i\n", rightHandSquare);
    //result = (leftHandSquare + rightHandSquare) / 2.0;

    //printf("Number of iterations: %i\n", numIterations);

    return result;
}

float AbsoluteValue(float val)
{
    return val > 0 ? val : -val;
}

float Factorial(uint32_t value)
{
    if( value == 0 )
        return 1;

    return value * Factorial(value - 1);
}

float NumericallyCalculateEulersNumber(int numSums)
{
    float eulersNumber = 0;

    for(int i = 0; i < numSums; i++)
    {
        eulersNumber += (1 / Factorial(i));
    }
    //printf("eulersNumber approximated to %i sums: %f\n", numSums, eulersNumber);

    return eulersNumber;
}

int round_down(float val)
{
    // implicit conversion to integer, causing loss of the decimal part
    return val;
}

void GrabVector(float** matrix, int numRows, int numCols, char* direction, int desiredIndex, float* out)
{
    if( !out )
    {
        printf("Result output must be initialized!\n");
        return;
    }

    /*printf("OK1\n"); CANT DO THIS BECAUSE direction is A CONSTANT STRING!
    int strLength = strlen(direction);
    printf("strlen = %i\n", strLength);
    for( int i = 0; i < strLength; i++ )
    {
        direction[i] = tolower(direction[i]);
        printf("direction[%i] = %c\n", i, direction[i]);
    }
    printf("OK2\n");*/

    if( direction == "row" )
    {
        if(desiredIndex >= numRows)
        {
            printf("desiredIndex is out of bounds!: %i\n", desiredIndex);
            return;
        }

        for(int i = 0 ; i < numRows; i++)
        {
            if( i == desiredIndex )
            {
                // Grab the desired row
                for(int j = 0; j < numCols; j++)
                {
                    out[j] = matrix[i][j];
                }
            }
        }
    }
    else if( direction == "column" || direction == "col" )
    {
        if(desiredIndex >= numCols)
        {
            printf("desiredIndex is out of bounds!: %i\n", desiredIndex);
            return;
        }

        for(int i = 0 ; i < numCols; i++)
        {
            if( i == desiredIndex )
            {
                // Grab the desired column
                for(int j = 0; j < numRows; j++)
                {
                    out[j] = matrix[j][i];
                }
            }
        }
    }
    else
    {
        printf("The direction must be either row or column!\n");
        printf("You gave: %s\n", direction);
        return;
    }
}

float StandardDeviation(float* featureValues, float mean, int sampleSize)
{
    if(sampleSize <= 0)
    {
        printf("Sample Size is not valid: %i\n", sampleSize);
        return -1;
    }

    float result = 0;

    for(int i = 0; i < sampleSize; i++)
    {
        //float value = Exponentiate(featureValues[i] - mean, 2);
        //printf("VALUE: %f\n", value);

        result += Exponentiate(featureValues[i] - mean, 2);
    }

    return SquareRoot(result / sampleSize);
}

void UserProvidedFeature(CostFunction* costFunction, TrainingSet* trainingSet)
{
    float* features = malloc(sizeof(float) * trainingSet->numFeatures);
    features[0] = 1;

    char anotherEstimate;

    do
    {
        for(int i = 1; i < trainingSet->numFeatures; i++)
        {
            printf("\nGive value to estimate for feature %i: ", i);
            scanf("%f", &features[i]);
        }
        printf("Guess is: %0.2f\n", LinearRegressionHypothesis(features, costFunction->parameters, costFunction->numParameters));

        do
        {
            fflush(stdin); // Flush stdin buffer
            printf("\nAnother estimate? (y/n): ");
            scanf("%c", &anotherEstimate);
            printf("\n");
        }while( anotherEstimate != 'y' && anotherEstimate != 'Y' && anotherEstimate != 'n' && anotherEstimate != 'N' );

    }while( anotherEstimate == 'y' || anotherEstimate == 'Y' );

    free(features);
}

void TrainWithLinearRegression(CostFunction* costFunction, TrainingSet* trainingSet)
{
    float* thetas = malloc(sizeof(float) * costFunction->numParameters);

    // Use a more compact variable name
    for(int i = 0; i < costFunction->numParameters; i++)
    {
        thetas[i] = costFunction->parameters[i];
    }

    float threashold = 0.0001;
    //float threashold = 0.01;
    //float precision = 10000;
    float precision = 10000;
    
    uint8_t converged;
    int iterations = 0;

    float* theta_sums = malloc(sizeof(float) * costFunction->numParameters);
    float* theta_adjustments = malloc(sizeof(float) * costFunction->numParameters);

    do
    {
        iterations++;

        // Find the sum that is used for each of the parameters
        for(int k = 0; k < costFunction->numParameters; k++)
        {
            theta_sums[k] = 0;
            // Iterate over each of the training examples
            for(int i = 0; i < trainingSet->size; i++)
            {
                float estimated_value = LinearRegressionHypothesis(trainingSet->features[i], thetas, costFunction->numParameters);
                // TODO: Look at estimated value and thetasums when large data set with feature scaling(,,0)
                // TODO: NOte that we are subtraction our large target from our small scaled feature value...
                //printf("Estimated Value: %f\n", estimated_value);

                theta_sums[k] += (estimated_value - trainingSet->targets[i]) * trainingSet->features[i][k];

                //printf("Feature[%i][%i] = %f\n", i, k, trainingSet->features[i][k]);
            }
        }

        converged = 1; // Assume conversion. If possible, prove otherwise below.
        for(int k = 0; k < costFunction->numParameters; k++)
        {
            //printf("ThetaSums[%i] = %f\n", k, theta_sums[k]);
            theta_adjustments[k] = (LEARNING_RATE * (theta_sums[k] / (float) trainingSet->size));
            //printf("Theta Adjustment[%i] = %f\n", k, theta_adjustments[k]);

            costFunction->parameters[k] = thetas[k] = thetas[k] - theta_adjustments[k];
            //printf("Theta[%i] = %f\n", k, thetas[k]);

            theta_adjustments[k] = round_down( (theta_adjustments[k] * precision) + 0.5 ) / precision;
            //printf("Adjusted Theta Adjustment[%i] = %f\tThreashold = %f\n", k, theta_adjustments[k], threashold);
            //printf("Absolute(ThetaAdjustment[%i]) = %f\n", k, AbsoluteValue(theta_adjustments[k]));

            // If we have made a theta adjustment that is greater than our threashold, then we haven't converged yet, i.e. keep iterating
            if(AbsoluteValue(theta_adjustments[k]) > threashold)
            {
                converged = 0;
            }
            else
            {
                //printf("%f <= %f\n", AbsoluteValue(theta_adjustments[k]), threashold);
            }
        }

        float costThisIteration = RunLinearRegressionCostFunction(costFunction, trainingSet);
        if( costThisIteration < 0 )
        {
            printf("Cost function is negative! Meaning it overshot a spot where it should have ended!\n");
            printf("CostFunction at iteration %i = %f\n", iterations, costThisIteration);
            getchar();
        }
        printf("CostFunction at iteration %i = %f\n", iterations, costThisIteration);

    } while(!converged); // && RunLinearRegressionCostFunction(costFunction, trainingSet) > threashold);
    printf("Final CostFunction = %f\n", RunLinearRegressionCostFunction(costFunction, trainingSet));

    printf("There are %i parameters\n", costFunction->numParameters);
    printf("%i iterations to obtain: ", iterations);

    for(int i = 0; i < costFunction->numParameters; i++)
    {
        printf("%0.2f", thetas[i]);

        if(i > 0)
            printf("x^%i", i);
        printf("  ");

        if( i != costFunction->numParameters - 1 )
            printf("+  ");
    }
    printf("\n");

    free(thetas);
    free(theta_sums);
    free(theta_adjustments);
}


int main(int argc, char** argv)
{
    srand(time(NULL));
    EULERS_NUMBER = NumericallyCalculateEulersNumber(6);

    /*printf("Logarithm = %f\n", Logarithm(128, 10));
    exit(1);*/

    TrainingSet* trainingSet = CreateTrainingSet();
    //PrintTrainingSet(trainingSet);
    //ApplyFeatureScaling(trainingSet, 0);
    //printf("Normalized Training Set\n-------------------------------------------\n");
    PrintTrainingSet(trainingSet);

    CostFunction* costFunction = CreateCostFunction(trainingSet->numFeatures);

    TrainWithLinearRegression(costFunction, trainingSet);

    UserProvidedFeature(costFunction, trainingSet);

    DestroyCostFunction(costFunction);
    DestroyTrainingSet(trainingSet);

    return 0;
}


