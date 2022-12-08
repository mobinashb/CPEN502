package LUTNN;

import LUT.Action;
import LUT.LUTUtils;
import NN.NeuralNet;

import java.io.*;
import java.util.Arrays;

public class LUTNN {
    public static final String FILENAME = "./data.csv";
    public static void main(String[] args) {
        double totalError;
        double errorThreshold = 0.07;

        int maxTrainSet = 112000;
        int numTrainSet = 0;
        final int NUM_FEATURES = 14;
        double [][] trainInput = new double[maxTrainSet][NUM_FEATURES];
        double [] trainOutput = new double[maxTrainSet];

        double learningRate, momentumTerm;
        int numHidden;
        int numTrial = 1;
        learningRate = 0.001;
        momentumTerm = 0.9;
        numHidden = 28;

//        try {
//            numTrainSet = load(trainInput, trainOutput);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        NeuralNet lutNN = new NeuralNet(14, 1, numHidden, learningRate, momentumTerm, -1, 1, -0.5, 0.5);
//        lutNN.initializeWeights();
//
//        for (int t = 0; t < numTrial; t++) {
//
//            double RMSError = 1;
//            while (RMSError > errorThreshold){
//                totalError = 0;
//
//                for (int i = 0; i < numTrainSet; i++) {
//                    totalError += Math.pow(lutNN.getError(trainInput[i], trainOutput[i]), 2);
//                }
//
//                RMSError = Math.sqrt(totalError / numTrainSet);
//                System.out.println(RMSError);
//
//                for (int i = 0; i < numTrainSet; i++) {
//                    lutNN.train(trainInput[i], trainOutput[i]);
//                }
//            }
//        }
        double[] testOneHot = LUTUtils.oneHotEncoding(6, 7);
        double[] stateAction = getStateActionPair(testOneHot, 2);
        System.out.println(Arrays.toString(stateAction));
    }

    public static double[] getStateActionPair(double[] state, int action) {
        double[] oneHotAction = LUTUtils.oneHotEncoding(action, Action.ROBOT_NUM_ACTIONS);

        int sLen = state.length;
        int aLen = oneHotAction.length;

        double[] stateAction = new double[sLen + aLen];
        for (int i = 0; i < sLen; i++)
            stateAction[i] = state[i];
        for (int i = 0; i < aLen; i++)
            stateAction[sLen + i] = oneHotAction[i];

        return stateAction;
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
