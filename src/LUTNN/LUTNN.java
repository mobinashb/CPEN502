package LUTNN;

import NN.NeuralNet;

import java.io.*;

public class LUTNN {
    public static final String FILENAME = "/Users/mobina/Desktop/UBC/Fall2022/CPEN502/CPEN502/data.csv";
    public static void main(String[] args) {
        double totalError;
        double errorThreshold = 0.05;

        int maxTrainSet = 112000;
        int numTrainSet = 0;
        final int NUM_FEATURES = 14;
        double [][] trainInput = new double[maxTrainSet][NUM_FEATURES];
        double [] trainOutput = new double[maxTrainSet];

        double learningRate, momentumTerm;
        int numHidden;
        int numTrial = 1;
        int epoch = 0;
        learningRate = 0.2;
        momentumTerm = 0.9;
        numHidden = 28;

        try {
            numTrainSet = load(trainInput, trainOutput);
        } catch (IOException e) {
            e.printStackTrace();
        }

        NeuralNet lutNN = new NeuralNet(14, numHidden,1,learningRate, momentumTerm, -1, 1);

        for (int t = 0; t < numTrial; t++) {
            lutNN.initializeWeights();
            lutNN.initializeTrainSet(trainInput, convertTo2d(trainOutput));
            epoch = 0;
            lutNN.train();

//            double RMSError;
//            do {
//                totalError = 0;
//                epoch++;
//
//                for (int i = 0; i < numTrainSet; i++) {
//                    totalError += Math.pow(lutNN.train(trainInput[i], trainOutput[i]), 2);
//                }
//                RMSError = Math.sqrt(totalError / numTrainSet);
//                System.out.println(RMSError);
//            } while (RMSError > errorThreshold);
        }
    }

    public static double[][] convertTo2d(double[] array) {
        double[][] res = new double[array.length][1];
        for (int i = 0; i < array.length; i++) {
            res[i] = new double[]{array[i]};
        }
        return res;
    }

    /**
     * Load LUT file (from assignment part 2) into training data set
     * @param trainInput array of training input.
     * @param trainOutput array of training output.
     * @return number of training data.
     */
    public static int load(double [][] trainInput, double [] trainOutput) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(FILENAME));
        String line = reader.readLine(); // skipping the header line
        int i = 0;
        double minQ = Double.MAX_VALUE;
        double maxQ = Double.MIN_VALUE;

        try {
            while ((line = reader.readLine()) != null) {
                String[] splitLine = line.split(",");
                for (int j = 0; j < splitLine.length - 1; j++) {
                    trainInput[i][j] = Double.parseDouble(splitLine[j]);
                }
                trainOutput[i] = Double.parseDouble(splitLine[splitLine.length - 1]);
                if (trainOutput[i] < minQ) {
                    minQ = trainOutput[i];
                }
                if (trainOutput[i] > maxQ) {
                    maxQ = trainOutput[i];
                }
                i++;
            }

            // normalize the Q values to (-1, 1)
            for (int k = 0; k < trainInput.length; k++) {
                trainOutput[k] = (trainOutput[k] - minQ) * 2 / (maxQ - minQ) - 1;
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            reader.close();
        }

        return i;
    }
}
