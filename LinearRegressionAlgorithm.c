
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <stdint.h>

#define USE_HARD_CODED_DATA 1

#define NUMBER_TRAINING_EXAMPLES 4
#define MAX_DATA_RANGE 10
#define MIN_DATA_RANGE 0

#define NUMBER_PARAMETERS 2

#define LEARNING_RATE .1 // 0.5 seems to be a sweet spot? 0.6 over takes much longer than anything yet

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
    int num_parameters;
    float* parameters;
} CostFunction;

// TODO: SOmething is up with numFeatures -> or maybe just everything is broken.
// TODO: Trying to support multiple features............
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
    int numTrainingSamplesRead = 0;
    int data = 0;

    while( (ch = fgetc(fp)) != EOF && ch != '-' )
    {
        // Just eat up everything before the "-------" section of the file (used for notes and such)
    }

    trainingSet->features[trainingSet->size] = malloc(sizeof(float) * 1);

    while( (ch = fgetc(fp)) != EOF )
    {
        if( onFeature )
        {
            if( isdigit(ch) )
            {
                data = (data * 10) + (ch - '0');
            }
            else if(ch == ',' || ch == ')')
            {
                trainingSet->features[trainingSet->size] = realloc(trainingSet->features[trainingSet->size], 
                                                                    sizeof(float) * (trainingSet->numFeatures + 1));
                trainingSet->features[trainingSet->size][trainingSet->numFeatures++] = data;
                data = 0;
            }
            else if( ch == '\n' )
            {
                onFeature = 0;
                trainingSet->numFeatures = 0;
            }
        }
        else
        {
            if( isdigit(ch) )
            {
                data = (data * 10) + (ch - '0');
            }
            else if(ch == ',' || ch == ')')
            {
                trainingSet->targets = realloc(trainingSet->targets, sizeof(float) * (trainingSet->numTargets + 1));
                trainingSet->targets[trainingSet->numTargets++] = data;
                data = 0;
            }
            else if( ch == '\n' )
            {
                //trainingSet->size++;
                trainingSet->features[++trainingSet->size] = malloc(sizeof(float) * 1);
                onFeature = 1;
            }
        }
    }
    printf("Training set size: %i\n", trainingSet->size);


    fclose(fp);
}

TrainingSet* CreateTrainingSet()
{
    TrainingSet* trainingSet = malloc(sizeof(trainingSet));

    trainingSet->size = 0; //NUMBER_TRAINING_EXAMPLES;
    trainingSet->numFeatures = 0;
    trainingSet->numTargets = 0;

    trainingSet->features = malloc(sizeof(float*) * 1);
    trainingSet->targets = malloc(sizeof(float) * 1);

    ReadTrainingData(trainingSet);

    /*if( !USE_HARD_CODED_DATA )
    {
        for(int i = 0; i < trainingSet->size; i++)
        {
            trainingSet->features[i] = ( (rand() % (MAX_DATA_RANGE - MIN_DATA_RANGE)) + MIN_DATA_RANGE );
            trainingSet->targets[i] = ( (rand() % (MAX_DATA_RANGE - MIN_DATA_RANGE)) + MIN_DATA_RANGE );
        }
    }
    else
    {
        // TODO: Make this pretier (So I can initialize a 2D array on 1 line)
        trainingSet->features[0] = 1;
        trainingSet->targets[0] = 0.5;
        trainingSet->features[1] = 2;
        trainingSet->targets[1] = 1;
        trainingSet->features[2] = 4;
        trainingSet->targets[2] = 2;
        trainingSet->features[3] = 0;
        trainingSet->targets[3] = 0;
    }*/

    return trainingSet;
}

void DestroyTrainingSet(TrainingSet* trainingSet)
{
    free(trainingSet->features);
    free(trainingSet->targets);
    free(trainingSet);
}

void PrintTrainingSet(TrainingSet* trainingSet)
{
    for(int i = 0; i < trainingSet->size; i++)
    {
        printf("Features: ");
        for(int j = 0; j < trainingSet->numFeatures; j++)
        {
            printf("%0.2f ", trainingSet->features[i][j]);
            
            if(j != trainingSet->numFeatures - 1)
                printf(",");
            else
                printf("\n");
        }
        printf("Target(s): ");
        for(int j = 0; j < trainingSet->numTargets; j++)
        {
            printf("%0.2f ", trainingSet->targets[j]);
            
            if(j != trainingSet->numTargets - 1)
                printf(",");
        }
    }

    printf("\n");
}

CostFunction* CreateCostFunction()
{
    CostFunction* costFunction = malloc(sizeof(CostFunction));
    
    costFunction->num_parameters = NUMBER_PARAMETERS;
    costFunction->parameters = malloc(sizeof(float) * costFunction->num_parameters);

    for(int i = 0; i < costFunction->num_parameters; i++)
    {
        // Some random start point for our thetas (TODO: Assuming the need for init to be within domain of training set)
        costFunction->parameters[i] = ( (rand() % (MAX_DATA_RANGE - MIN_DATA_RANGE)) + MIN_DATA_RANGE );
    }

    return costFunction;
}

void DestroyCostFunction(CostFunction* costFunction)
{
    free(costFunction);
}

// Predict target (y value) given a feature (x value)
// Using "theta_0 + theta_1*x" as hypothesis function
float Hypothesis(float feature, float theta_0, float theta_1)
{
    /* TODO: May want to be more flexible in the future, the below may help
    int hypothesisSum = 0;
    for(int j = 0; j < costFunction->num_parameters; j++)
    {
        hypothesisSum += costFunction->parameters[j] * Exponentiate(trainingSet->features[i], j);
    }*/

    return theta_0 + theta_1*feature;
}

int Exponentiate(int base, int power)
{
    int result = 1;
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

void TrainWithLinearRegression(TrainingSet* trainingSet)
{
    CostFunction* costFunction = CreateCostFunction();

    // TODO: This is dependent on there being at least 2 parameters for our cost function
    float theta_0 = costFunction->parameters[0];
    float theta_1 = costFunction->parameters[1];

    float threashold = 0.001;
    float precision = 10000;
    
    float theta_0_step_adjustment;
    float theta_1_step_adjustment;

    int iterations = 0;

    do
    {
        float theta_0_sum = 0;
        float theta_1_sum = 0;
        for(int i = 0; i < trainingSet->size; i++)
        {
            float estimated_value = 0;// TODO:: Hypothesis(trainingSet->features[i], theta_0, theta_1);
            theta_0_sum += estimated_value - trainingSet->targets[i];
            theta_1_sum += (estimated_value - trainingSet->targets[i]) * trainingSet->targets[i];
        }

        theta_0_step_adjustment = (LEARNING_RATE * (theta_0_sum / (float) trainingSet->size));
        theta_1_step_adjustment = (LEARNING_RATE * (theta_1_sum / (float) trainingSet->size));

        theta_0 = theta_0 - theta_0_step_adjustment;
        theta_1 = theta_1 - theta_1_step_adjustment;

        theta_0_step_adjustment = round_down( (theta_0_step_adjustment * precision) + 0.5 ) / precision;
        theta_1_step_adjustment = round_down( (theta_1_step_adjustment * precision) + 0.5 ) / precision;

        iterations++;
    } while(AbsoluteValue(theta_0_step_adjustment) > threashold || 
            AbsoluteValue(theta_1_step_adjustment) > threashold);

    printf("%i iterations to obtain: %0.2f + %0.2fx\n", iterations, theta_0, theta_1);

    float feature;
    printf("Give value to estimate: ");
    scanf("%f", &feature);

    printf("Guess is: %0.2f\n", Hypothesis(feature, theta_0, theta_1));

    DestroyCostFunction(costFunction);
}

int main(int argc, char** argv)
{
    srand(time(NULL));

    TrainingSet* trainingSet = CreateTrainingSet();

    //ReadTrainingData(trainingSet);
    PrintTrainingSet(trainingSet);
    //TrainWithLinearRegression(trainingSet);

    DestroyTrainingSet(trainingSet);

    return 0;
}


