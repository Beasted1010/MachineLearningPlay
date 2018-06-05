
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <stdint.h>

#define MAX_DATA_RANGE 10
#define MIN_DATA_RANGE 0

#define LEARNING_RATE .00000005
//#define LEARNING_RATE .3 // For 1 feature training data

#define TRAINING_DATA_FILE "training_data.txt"

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
float Hypothesis(float* features, float* thetas, int numParameters);
void PrintTrainingSet(TrainingSet* trainingSet);

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
// TODO: This may not be correct, results seem to not be val / min-max ???? May want to disect
void ApplyFeatureScaling(TrainingSet* trainingSet, uint8_t applyMeanNormalization)
{
    float* maxFeatureVal = malloc(sizeof(float) * trainingSet->numFeatures);
    float* minFeatureVal = malloc(sizeof(float) * trainingSet->numFeatures);

    float* featureValRange = malloc(sizeof(float) * trainingSet->numFeatures);

    float* featureValSum = malloc(sizeof(float) * trainingSet->numFeatures);

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

            featureValSum[featureIndex] += trainingSet->features[trainingExampleIndex][featureIndex];
        }

        printf("Max feature value: %f\tMin feature value: %f\n", maxFeatureVal[featureIndex], minFeatureVal[featureIndex]);
        featureValRange[featureIndex] = maxFeatureVal[featureIndex] - minFeatureVal[featureIndex];
        printf("featureValRange[%i] = %f\n", featureIndex, featureValRange[featureIndex]);
    }

    for( int i = 0; i < trainingSet->size; i++ )
    {
        float averageFeatureVal;
        if(applyMeanNormalization)
            averageFeatureVal = featureValSum[i] / trainingSet->size;
        printf("Feature val sum: %f\n", featureValSum[i]);
        printf("Average feature val: %f\n", averageFeatureVal);

        for( int j = 0; j < trainingSet->numFeatures; j++ )
        {
            printf("Feature value range = %f\n", featureValRange[j]);
            printf("Feature %i before: %f\t", j, trainingSet->features[i][j]);
            // Scale the feature down between a range (0 - 1)
            trainingSet->features[i][j] /= featureValRange[j];

            if(applyMeanNormalization)
                trainingSet->features[i][j] -= (averageFeatureVal / featureValRange[j]);
            else // This is in reference to wikipedia's version of feature scaling by mean sof "rescaling"
                trainingSet->features[i][j]; // -= minFeatureVal[j]; Assuming I don't want this, since negatives?
            printf("Feature %i after: %f\n", j, trainingSet->features[i][j]);
        }
    }

    printf("Normalized TrainingSet\n-----------------------\n");
    PrintTrainingSet(trainingSet);

    free(maxFeatureVal);
    free(minFeatureVal);
    free(featureValRange);
    free(featureValSum);
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

    for(int i = 0; i < costFunction->numParameters; i++)
    {
        // Some random start point for our thetas (TODO: Assuming the need for init to be within domain of training set) which isn't true for my recent training sets...
        // TODO: Make this more dynamic and scalable for MAX_DATA_RANGE and MIN_... depending on max/min of data?
        costFunction->parameters[i] = ( (rand() % (MAX_DATA_RANGE - MIN_DATA_RANGE)) + MIN_DATA_RANGE );
    }

    return costFunction;
}

void DestroyCostFunction(CostFunction* costFunction)
{
    free(costFunction->parameters);
    free(costFunction);
}

float RunCostFunction(CostFunction* costFunction, TrainingSet* trainingSet)
{
    float cost_function_sum = 0;

    // Find the sum that is used for each of the parameters
    for(int i = 0; i < trainingSet->size; i++)
    {
        float estimated_value = Hypothesis(trainingSet->features[i], costFunction->parameters, costFunction->numParameters);
        cost_function_sum += Exponentiate( (estimated_value - trainingSet->targets[i]), 2 );
    }

    return ( cost_function_sum / (float) (2.0 * trainingSet->size) );
}

// Predict target (y value) given a feature (x value)
// Using "theta_0 + theta_1*x" as hypothesis function
float Hypothesis(float* features, float* thetas, int numParameters)
{
    float hypothesisSum = 0;
    for(int i = 0; i < numParameters; i++)
    {
        hypothesisSum += thetas[i] * features[i];

        //printf("thetas[%i] = %f\tfeatures[%i] = %f\n", i, thetas[i], i, features[i]);
    }
    
    //printf("hyptohesisSum = %f\n", hypothesisSum);

    return hypothesisSum;
}

float Exponentiate(float base, int power)
{
    float result = 1;
    for(int i = 0; i < power; i++)
    {
        result *= base;
    }

    return result;
}

float AbsoluteValue(float val)
{
    return val > 0 ? val : -val;
}

int round_down(float val)
{
    // implicit conversion to integer, causing loss of the decimal part
    return val;
}

void UserProvidedFeature(CostFunction* costFunction, TrainingSet* trainingSet, float* thetas)
{
    float* features = malloc(sizeof(float) * trainingSet->numFeatures);
    features[0] = 1;
    for(int i = 1; i < trainingSet->numFeatures; i++)
    {
        printf("\nGive value to estimate for feature %i: ", i);
        scanf("%f", &features[i]);
    }

    printf("Guess is: %0.2f\n", Hypothesis(features, thetas, costFunction->numParameters));
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
                float estimated_value = Hypothesis(trainingSet->features[i], thetas, costFunction->numParameters);
                theta_sums[k] += (estimated_value - trainingSet->targets[i]) * trainingSet->features[i][k];

                //printf("Feature[%i][%i] = %f\n", i, k, trainingSet->features[i][k]);
            }
        }

        for(int k = 0; k < costFunction->numParameters; k++)
        {
            theta_adjustments[k] = (LEARNING_RATE * (theta_sums[k] / (float) trainingSet->size));

            costFunction->parameters[k] = thetas[k] = thetas[k] - theta_adjustments[k];

            theta_adjustments[k] = round_down( (theta_adjustments[k] * precision) + 0.5 ) / precision;

            converged = 1; // Assume conversion. If possible, prove otherwise below.
            // If we have made a theta adjustment that is greater than our threashold, then we haven't converged yet, i.e. keep iterating
            if(AbsoluteValue(theta_adjustments[k]) > threashold)
            {
                converged = 0;
            }
        }

        printf("CostFunction at iteration %i = %f\n", iterations, RunCostFunction(costFunction, trainingSet));

    } while(!converged);
    ApplyFeatureScaling(trainingSet, 0);

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

    UserProvidedFeature(costFunction, trainingSet, thetas);

    free(thetas);
    free(theta_sums);
    free(theta_adjustments);
}

int main(int argc, char** argv)
{
    srand(time(NULL));

    TrainingSet* trainingSet = CreateTrainingSet();
    CostFunction* costFunction = CreateCostFunction(trainingSet->numFeatures);

    PrintTrainingSet(trainingSet);
    TrainWithLinearRegression(costFunction, trainingSet);

    DestroyCostFunction(costFunction);
    DestroyTrainingSet(trainingSet);

    return 0;
}


