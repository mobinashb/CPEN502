package NN;

import Interfaces.NeuralNetInterface;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.Math;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;


public class NeuralNet implements NeuralNetInterface
{

    private static final int MAX_HIDDEN_NEURONS = 1000;
    private static final int MAX_INPUTS = 15;
    private static final int MAX_OUTPUTS = 7;



    private double argA;
    private double argB;
    private double weightInitMin;
    private double weightInitMax;




    private int numInputs;
    private int numOutputs;
    private int numHiddenNeurons;


    private double learningRate;
    private double momentumTerm;


    private double[] inputValues = new double[MAX_INPUTS];


    private static double[][] inputWeights = new double[MAX_HIDDEN_NEURONS][MAX_INPUTS];

    private static double[][] previousInputWeights = new double[MAX_HIDDEN_NEURONS][MAX_INPUTS];


    private double[] hiddenNeuronUnactivatedOutputs = new double[MAX_HIDDEN_NEURONS];

    private double[] hiddenNeuronOutputs = new double[MAX_HIDDEN_NEURONS];

    private double[] hiddenNeuronErrors = new double[MAX_HIDDEN_NEURONS];


    private static double[][] outputNeuronWeights = new double[MAX_OUTPUTS][MAX_HIDDEN_NEURONS];

    private static double[][] previousOutputNeuronWeights = new double[MAX_OUTPUTS][MAX_HIDDEN_NEURONS];

    private static double[] outputNeuronBiasWeights = new double[MAX_OUTPUTS];
    private static double[] previousOutputNeuronBiasWeights = new double[MAX_OUTPUTS];


    private double[] outputNeuronUnactivatedValues = new double[MAX_OUTPUTS];

    private double[] outputNeuronValues = new double[MAX_OUTPUTS];

    private double[] outputNeuronErrors = new double[MAX_OUTPUTS];

    
    public NeuralNet(int argNumInputs,
                          int argNumOutputs,
                          int argNumHidden,
                          double argLearningRate,
                          double argMomentumTerm,
                          double argA,
                          double argB,
                          double argWeightInitMin,
                          double argWeightInitMax)
    {

        this.argA = argA;
        this.argB = argB;
        weightInitMin = argWeightInitMin;
        weightInitMax = argWeightInitMax;

        numInputs = argNumInputs + 1;
        numOutputs = argNumOutputs;
        numHiddenNeurons = argNumHidden;

        learningRate = argLearningRate;
        momentumTerm = argMomentumTerm;

        zeroWeights();


    }

    
    public double sigmoid(double x)
    {
        double result;

        result =  1 / (1 + Math.exp(-x));

        return result;
    }

    
    public double sigmoidDerivative(double x)
    {
        double result;

        result = sigmoid(x)*(1 - sigmoid(x));

        return result;
    }

    
    public double customSigmoid(double x)
    {
        double result;

        result = (argB - argA) * sigmoid(x) + argA;

        return result;
    }

    
    public double customSigmoidDerivative(double x)
    {
        double result;

        result = (1.0/(argB - argA)) * (customSigmoid(x) - argA) * (argB - customSigmoid(x));

        return result;
    }

    
    public void initializeWeights()
    {
        int i, j;


        for(i = 0; i < numHiddenNeurons; i++)
        {
            for(j = 0; j < numInputs; j++)
            {
                inputWeights[i][j] = getRandomDouble(weightInitMin, weightInitMax);
            }
        }


        for(i = 0; i < numOutputs; i++)
        {
            for(j = 0; j < numHiddenNeurons; j++)
            {

                outputNeuronWeights[i][j] = getRandomDouble(weightInitMin, weightInitMax);
                outputNeuronBiasWeights[i] = getRandomDouble(weightInitMin, weightInitMax);
            }
        }


        previousInputWeights = inputWeights.clone();
        previousOutputNeuronWeights = outputNeuronWeights.clone();
        previousOutputNeuronBiasWeights = outputNeuronBiasWeights.clone();
    }

    private double calculateWeightDelta(double weightInput, double error, double currentWeight, double previousWeight)
    {
        double momentumTerm, learningTerm;

        momentumTerm = this.momentumTerm * (currentWeight - previousWeight);
        learningTerm = learningRate * error * weightInput;
        return (momentumTerm + learningTerm);
    }

    
    private void updateWeights()
    {
        int hiddenNeuron, outputNeuron, input;
        double[] newOutputNeuronBiasWeights = new double[MAX_OUTPUTS];
        double[][] newOutputNeuronWeights = new double[MAX_OUTPUTS][MAX_HIDDEN_NEURONS];
        double[][] newInputNeuronWeights = new double[MAX_HIDDEN_NEURONS][MAX_INPUTS];



        for(outputNeuron = 0; outputNeuron < numOutputs; outputNeuron++)
        {

            newOutputNeuronBiasWeights[outputNeuron] =
                    outputNeuronBiasWeights[outputNeuron] +
                            calculateWeightDelta(1.0,
                                    outputNeuronErrors[outputNeuron],
                                    outputNeuronBiasWeights[outputNeuron],
                                    previousOutputNeuronBiasWeights[outputNeuron]);


            for(hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++)
            {
                newOutputNeuronWeights[outputNeuron][hiddenNeuron] =
                        outputNeuronWeights[outputNeuron][hiddenNeuron] +
                                calculateWeightDelta(
                                        hiddenNeuronOutputs[hiddenNeuron],
                                        outputNeuronErrors[outputNeuron],
                                        outputNeuronWeights[outputNeuron][hiddenNeuron],
                                        previousOutputNeuronWeights[outputNeuron][hiddenNeuron]);
            }
        }


        for(hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++)
        {
            for(input = 0; input < numInputs; input++)
            {
                newInputNeuronWeights[hiddenNeuron][input] = inputWeights[hiddenNeuron][input] +
                        calculateWeightDelta(
                                inputValues[input],
                                hiddenNeuronErrors[hiddenNeuron],
                                inputWeights[hiddenNeuron][input],
                                previousInputWeights[hiddenNeuron][input]);
            }
        }

        previousOutputNeuronBiasWeights = outputNeuronBiasWeights.clone();
        previousOutputNeuronWeights = outputNeuronWeights.clone();
        previousInputWeights = inputWeights.clone();

        outputNeuronBiasWeights = newOutputNeuronBiasWeights.clone();
        outputNeuronWeights = newOutputNeuronWeights.clone();
        inputWeights = newInputNeuronWeights.clone();
    }

    
    private double getRandomDouble(double min, double max)
    {
        double random, result;

        random = new Random().nextDouble();
        result = min + (random * (max - min));

        return result;
    }

    public void zeroWeights()
    {
        int i, j;


        for(i = 0; i < numHiddenNeurons; i++)
        {
            for(j = 0; j < numInputs; j++)
            {
                inputWeights[i][j] = 0.0;
                previousInputWeights[i][j] = 0.0;
            }
        }


        for(i = 0; i < numOutputs; i++)
        {
            for(j = 0; j < numHiddenNeurons; j++)
            {
                previousOutputNeuronWeights[i][j] = 0.0;
                outputNeuronWeights[i][j] = 0.0;
            }
        }
    }

    
    public double outputFor(double[] x)
    {
        int hiddenNeuron, outputNeuron, input, index;

        inputValues[0] = 1.0;

        for (index = 0; index < x.length; index++)
        {
            inputValues[index+1] = x[index];
        }



        for(hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++)
        {
            hiddenNeuronUnactivatedOutputs[hiddenNeuron] = 0.0;

            for(input = 0; input < numInputs; input++)
            {
                hiddenNeuronUnactivatedOutputs[hiddenNeuron] += inputWeights[hiddenNeuron][input] * inputValues[input];
            }

            hiddenNeuronOutputs[hiddenNeuron] = customSigmoid(hiddenNeuronUnactivatedOutputs[hiddenNeuron]);
        }


        for(outputNeuron = 0; outputNeuron < numOutputs; outputNeuron++)
        {
            outputNeuronUnactivatedValues[outputNeuron] = 0.0;
            for(hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++)
            {
                outputNeuronUnactivatedValues[outputNeuron] += hiddenNeuronOutputs[hiddenNeuron] * outputNeuronWeights[outputNeuron][hiddenNeuron];
            }

            outputNeuronUnactivatedValues[outputNeuron] += (1.0 * outputNeuronBiasWeights[outputNeuron]);

            outputNeuronValues[outputNeuron] = customSigmoid(outputNeuronUnactivatedValues[outputNeuron]);
        }

        return outputNeuronValues[0];
    }

    
    private void calculateErrors(double expectedValue)
    {
        int hiddenNeuron, outputNeuron, outputNeuronIndex;
        double summedWeightedErrors;

        for(outputNeuron = 0; outputNeuron < numOutputs; outputNeuron++)
        {

            outputNeuronErrors[outputNeuron] = (expectedValue - outputNeuronValues[outputNeuron]) * customSigmoidDerivative(outputNeuronUnactivatedValues[outputNeuron]);


            for(hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++)
            {
                summedWeightedErrors = 0.0;


                for(outputNeuronIndex = 0; outputNeuronIndex < numOutputs; outputNeuronIndex++)
                {
                    summedWeightedErrors += outputNeuronErrors[outputNeuronIndex] * outputNeuronWeights[outputNeuronIndex][hiddenNeuron];
                }

                hiddenNeuronErrors[hiddenNeuron] = summedWeightedErrors * customSigmoidDerivative(hiddenNeuronUnactivatedOutputs[hiddenNeuron]);
            }
        }
    }

    
    public double train(double[] trainX, double trainY)
    {
        int i;
        double[] errors = new double[numOutputs];



        outputFor(trainX);


        calculateErrors(trainY);


        updateWeights();

        for(i = 0; i < numOutputs; i++)
        {
            errors[i] = trainY - outputNeuronValues[i];
        }


        return Arrays.stream(errors).sum();
    }

    public double getError(double[] trainX, double trainY) {
        double[] errors = new double[numOutputs];

        outputFor(trainX);
        calculateErrors(trainY);

        for(int i = 0; i < numOutputs; i++)
        {
            errors[i] = trainY - outputNeuronValues[i];
        }

        return Arrays.stream(errors).sum();
    }

    @Override
    public void save(File argFile) {
    }

    @Override
    public void load(String argFileName) throws IOException {
    }
}