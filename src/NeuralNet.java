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

    private static int numInputs = 2;
    private static int numHiddenNeurons = 4;
    private int numOutputs = 1;
    private double learningRate = 0.2;
    private double momentum = 0.0;
    private double a = 0;
    private double b = 1;
    private boolean isBinary = true;
    private double initRangeMax = 0.5;
    private double initRangeMin = -0.5;
    private static double errorThreshold = 0.05;
    private double[] inputLayer = new double[numInputs + 1];  //one extra bias node
    private double[] hiddenLayer = new double[numHiddenNeurons + 1];
    private double[] outputLayer = new double[numOutputs];
    private double[][] w1 = new double[numInputs + 1][numHiddenNeurons];
    private double[][] w2 = new double[numHiddenNeurons + 1][numOutputs];
    private double[] deltaOutput = new double[numOutputs];
    private double[] deltaHidden = new double[numHiddenNeurons];
    private double[][] deltaW1 = new double[numInputs + 1][numHiddenNeurons];
    private double[][] deltaW2 = new double[numHiddenNeurons + 1][numOutputs];
    private double[] totalError = new double[numOutputs];
    private double[] singleError = new double[numOutputs];
    private List<String> error = new LinkedList<>();
    private double[][] trainX;
    private double[][] trainY;

    public NeuralNet(int numInputs, int numHiddenNeurons, int numOutputs,
                     double learningRate, double momentum,
                     double a, double b) {
        this.numInputs = numInputs;
        this.numHiddenNeurons = numHiddenNeurons;
        this.numOutputs = numOutputs;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.a = a;
        this.b = b;
    }

    public NeuralNet() {
    }

    public void setRepresentation(boolean isBinary) {
        this.isBinary = isBinary;
        if (!isBinary) {
            b = 1;
            a = -1;
        }
    }

    public void initializeTrainSet() {
        if (isBinary) {
            trainX = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
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
        return (b - a) / (1 + Math.exp(-x)) + a;
    }

    @Override
    public void initializeWeights() {
        Random r = new Random();
        for (int i = 0; i < numInputs + 1; i++) {
            for (int j = 0; j < numHiddenNeurons; j++) {
                w1[i][j] = initRangeMin + (initRangeMax - initRangeMin) * r.nextDouble();
                deltaW1[i][j] = 0;
            }
        }

        for (int j = 0; j < numHiddenNeurons + 1; j++) {
            for (int k = 0; k < numOutputs; k++) {
                w2[j][k] = initRangeMin + (initRangeMax - initRangeMin) * r.nextDouble();
                deltaW2[j][k] = 0;
            }
        }
    }

    @Override
    public void zeroWeights() {}

    private void initializeLayers(double[] sample) {
        for (int i = 0; i < numInputs; i++) {
            inputLayer[i] = sample[i];
        }
        inputLayer[numInputs] = 1;
        hiddenLayer[numHiddenNeurons] = 1;
    }

    private void forwardPropagation(double[] sample) {
        initializeLayers(sample);
        for (int j = 0; j < numHiddenNeurons; j++) {
            hiddenLayer[j] = 0;
            for (int i = 0; i < numInputs + 1; i++) {
                hiddenLayer[j] += w1[i][j] * inputLayer[i];
            }
            hiddenLayer[j] = customSigmoid(hiddenLayer[j]);
        }

        for (int k = 0; k < numOutputs; k++) {
            outputLayer[k] = 0;
            for (int j = 0; j < numHiddenNeurons + 1; j++) {
                outputLayer[k] += w2[j][k] * hiddenLayer[j];
            }
            outputLayer[k] = customSigmoid(outputLayer[k]);
        }
    }

    private void backPropagation() {
        for (int k = 0; k < numOutputs; k++) {
            deltaOutput[k] = 0;
            deltaOutput[k] = isBinary ? singleError[k] * outputLayer[k] * (1 - outputLayer[k]) :
                    singleError[k] * (outputLayer[k] + 1) * 0.5 * (1 - outputLayer[k]);
        }

        for (int k = 0; k < numOutputs; k++) {
            for (int j = 0; j < numHiddenNeurons + 1; j++) {
                deltaW2[j][k] = momentum * deltaW2[j][k] + learningRate * deltaOutput[k] * hiddenLayer[j];
                w2[j][k] += deltaW2[j][k];
            }
        }

        for (int j = 0; j < numHiddenNeurons; j++) {
            deltaHidden[j] = 0;
            for (int k = 0; k < numOutputs; k++) {
                deltaHidden[j] += w2[j][k] * deltaOutput[k];
            }
            deltaHidden[j] = isBinary ? deltaHidden[j] * hiddenLayer[j] * (1 - hiddenLayer[j]) :
                    deltaHidden[j] * (hiddenLayer[j] + 1) * 0.5 * (1 - hiddenLayer[j]);
        }

        for (int j = 0; j < numHiddenNeurons; j++) {
            for (int i = 0; i < numInputs + 1; i++) {
                deltaW1[i][j] = momentum * deltaW1[i][j]
                        + learningRate * deltaHidden[j] * inputLayer[i];
                w1[i][j] += deltaW1[i][j];
            }
        }

    }

    public int train() {
        int epoch = 0;
        error.clear();

        do {
            for (int k = 0; k < numOutputs; k++) {
                totalError[k] = 0;
            }
            int numSamples = trainX.length;
            for (int i = 0; i < numSamples; i++) {
                double[] sample = trainX[i];
                forwardPropagation(sample);
                for (int k = 0; k < numOutputs; k++) {
                    singleError[k] = trainY[i][k] - outputLayer[k];
                    totalError[k] += Math.pow(singleError[k], 2);
                }
                backPropagation();
            }

            for (int k = 0; k < numOutputs; k++) totalError[k] /= 2;
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
        Scanner sc = new Scanner(new BufferedReader(new FileReader("./weights.txt")));
        double[][] w1 = new double[numInputs + 1][numHiddenNeurons];
        double[][] w2 = new double[numHiddenNeurons + 1][numOutputs];
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

    public void saveError() {
        try {
            Files.write(Paths.get("./trainError.txt"), error);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}