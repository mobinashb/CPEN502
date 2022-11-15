package NN;

import Interfaces.NeuralNetInterface;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class NeuralNet implements NeuralNetInterface {

    private int argNumInputs = 2;
    private int argNumHidden = 4;
    private int numOfOutputs = 1;
    private double learningRate = 0.2;
    private double momentum = 0.0;
    private double argA = 0;
    private double argB = 1;
    private double initWeightsMax = 0.5;
    private double initWeightsMin = -0.5;
    private double errorThreshold = 0.05;

    private boolean isBinary = true; //binary or bipolar

    private double[] inputLayer = new double[argNumInputs + 1];  //one extra for the bias value
    private double[] hiddenLayer = new double[argNumHidden + 1];
    private double[] outputLayer = new double[numOfOutputs];

    private double[][] w1 = new double[argNumInputs + 1][argNumHidden];
    private double[][] w2 = new double[argNumHidden + 1][numOfOutputs];
    private double[][] deltaW1 = new double[argNumInputs + 1][argNumHidden];
    private double[][] deltaW2 = new double[argNumHidden + 1][numOfOutputs];
    private double[] deltaOutput = new double[numOfOutputs];
    private double[] deltaHidden = new double[argNumHidden];

    private double[] totalError = new double[numOfOutputs];
    private double[] singleError = new double[numOfOutputs];
    private List<String> error = new LinkedList<>();
    private double[][] trainX;
    private double[][] trainY;

    public NeuralNet() {}

    public NeuralNet(int argNumInputs, int argNumHidden, int numOfOutputs,
                     double learningRate, double momentum,
                     double argA, double argB) {
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.numOfOutputs = numOfOutputs;

        this.learningRate = learningRate;
        this.momentum = momentum;

        this.argA = argA;
        this.argB = argB;
    }

    public void setRepresentation(boolean isBinary) { //binary or bipolar
        this.isBinary = isBinary;
        if (!isBinary) {
            argB = 1;
            argA = -1;
        }
    }

    public void initializeTrainSet() {
        if (isBinary) {
            trainX = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}}; //xor
            trainY = new double[][]{{0}, {1}, {1}, {0}};
        } else {
            trainX = new double[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
            trainY = new double[][]{{-1}, {1}, {1}, {-1}};
        }
    }

    @Override
    public double sigmoid(double x) {
        return 2 / (1 + Math.exp(-x)) - 1;
    }

    @Override
    public double customSigmoid(double x) {
        return (argB - argA) / (1 + Math.exp(-x)) - (-1 * argA);
    }

    @Override
    public void initializeWeights() {
        Random r = new Random();
        for (int i = 0; i <= argNumInputs; i++) {
            for (int j = 0; j < argNumHidden; j++) {
                w1[i][j] = initWeightsMin + (initWeightsMax - initWeightsMin) * r.nextDouble();
                deltaW1[i][j] = 0;
            }
        }

        for (int j = 0; j <= argNumHidden; j++) {
            for (int k = 0; k < numOfOutputs; k++) {
                w2[j][k] = initWeightsMin + (initWeightsMax - initWeightsMin) * r.nextDouble();
                deltaW2[j][k] = 0;
            }
        }
    }

    @Override
    public void zeroWeights() {
        for (int i = 0; i <= argNumInputs; i++) {
            for (int j = 0; j < argNumHidden; j++) {
                w1[i][j] = 0;
                deltaW1[i][j] = 0;
            }
        }

        for (int j = 0; j <= argNumHidden; j++) {
            for (int k = 0; k < numOfOutputs; k++) {
                w2[j][k] = 0;
                deltaW2[j][k] = 0;
            }
        }
    }

    private void initializeLayers(double[] sample) {
        for (int i = 0; i < argNumInputs; i++) {
            inputLayer[i] = sample[i];
        }
        inputLayer[argNumInputs] = bias;
        hiddenLayer[argNumHidden] = bias;
    }

    private void forwardPropagation(double[] sample) {
        initializeLayers(sample);

        for (int j = 0; j < argNumHidden; j++) {
            hiddenLayer[j] = 0;
            for (int i = 0; i <= argNumInputs; i++) {
                hiddenLayer[j] += w1[i][j] * inputLayer[i];
            }
            hiddenLayer[j] = customSigmoid(hiddenLayer[j]);
        }

        for (int k = 0; k < numOfOutputs; k++) {
            outputLayer[k] = 0;
            for (int j = 0; j <= argNumHidden; j++) {
                outputLayer[k] += w2[j][k] * hiddenLayer[j];
            }
            outputLayer[k] = customSigmoid(outputLayer[k]);
        }
    }

    private void calculateOutputErrors(double[] sampleOutput) {
        for (int k = 0; k < numOfOutputs; k++) {
            singleError[k] = sampleOutput[k] - outputLayer[k];
            totalError[k] += Math.pow(singleError[k], 2);
        }
    }

    private void backPropagation() {
        for (int k = 0; k < numOfOutputs; k++) {
            deltaOutput[k] = isBinary ? singleError[k] * outputLayer[k] * (1 - outputLayer[k]) :
                    singleError[k] * 0.5 * (1 - (outputLayer[k] * outputLayer[k]));
        }

        for (int k = 0; k < numOfOutputs; k++) {
            for (int j = 0; j <= argNumHidden; j++) {
                deltaW2[j][k] = momentum * deltaW2[j][k] + learningRate * deltaOutput[k] * hiddenLayer[j];
                w2[j][k] += deltaW2[j][k];
            }
        }

        for (int j = 0; j < argNumHidden; j++) {
            deltaHidden[j] = 0;
            for (int k = 0; k < numOfOutputs; k++) {
                deltaHidden[j] += w2[j][k] * deltaOutput[k];
            }
            deltaHidden[j] = isBinary ? deltaHidden[j] * hiddenLayer[j] * (1 - hiddenLayer[j]) :
                    deltaHidden[j] * 0.5 * (1 - (hiddenLayer[j] * hiddenLayer[j]));
        }

        for (int j = 0; j < argNumHidden; j++) {
            for (int i = 0; i < argNumInputs + 1; i++) {
                deltaW1[i][j] = momentum * deltaW1[i][j] + learningRate * deltaHidden[j] * inputLayer[i];
                w1[i][j] += deltaW1[i][j];
            }
        }

    }

    public int train() {
        int epoch = 0;
        error.clear();

        do {
            for (int k = 0; k < numOfOutputs; k++) {
                totalError[k] = 0;
            }
            int numSamples = trainX.length;
            for (int i = 0; i < numSamples; i++) {
                double[] sample = trainX[i];
                double[] sampleOutput = trainY[i];

                forwardPropagation(sample);
                calculateOutputErrors(sampleOutput);
                backPropagation();
            }

            for (int k = 0; k < numOfOutputs; k++) totalError[k] /= 2;
            error.add(Double.toString(totalError[0]));
            epoch++;
        } while (totalError[0] > errorThreshold);

        return epoch;
    }

    @Override
    public double outputFor(double[] X) {
        forwardPropagation(X);
        return outputLayer[0];
    }

    @Override
    public double train(double[] X, double argValue) {
        forwardPropagation(X);
        return argValue - outputLayer[0];
    }

    @Override
    public void save(File argFile) {
        try {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < w1.length; i++) {
                for (int j = 0; j < w1[0].length; j++) {
                    builder.append(w1[i][j] + " ");
                }
                builder.append("\n");
            }
            builder.append("\n");
            for (int i = 0; i < w2.length; i++) {
                for (int j = 0; j < w2[0].length; j++) {
                    builder.append(w2[i][j] + " ");
                }
                builder.append("\n");
            }
            Files.write(argFile.toPath(), builder.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void load(String argFileName) throws IOException {
        Scanner sc = new Scanner(new BufferedReader(new FileReader(argFileName)));
        double[][] w1 = new double[argNumInputs + 1][argNumHidden];
        double[][] w2 = new double[argNumHidden + 1][numOfOutputs];
        boolean readingW1 = true;
        int lineIndex = 0;
        while (sc.hasNextLine()) {
            if (readingW1) {
                String[] line = sc.nextLine().trim().split(" ");
                if (line[0].length() == 0) {
                    readingW1 = false;
                    lineIndex = 0;
                    continue;
                }
                for (int j = 0; j < line.length; j++) {
                    w1[lineIndex][j] = Double.parseDouble(line[j]);
                }
                lineIndex++;
            } else {
                String[] line = sc.nextLine().trim().split(" ");
                if (line[0].length() == 0) {
                    break;
                }
                for (int j = 0; j < line.length; j++) {
                    w2[lineIndex][j] = Double.parseDouble(line[j]);
                }
                lineIndex++;
            }
        }
    }

    public void saveError(File argFile) {
        try {
            Files.write(Paths.get(argFile.getPath()), error);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}