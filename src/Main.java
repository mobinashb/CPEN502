import NN.NeuralNet;

import java.io.File;

public class Main {

    public static void main(String[] args) {
        // part a
//        NN.NeuralNet neuralNet = new NN.NeuralNet();
//        neuralNet.setRepresentation(true);
        // part b
//        NN.NeuralNet neuralNet = new NN.NeuralNet();
//        neuralNet.setRepresentation(false);
        // part c
        NeuralNet neuralNet = new NeuralNet(2, 4, 1, 0.2, 0.9, -1, 1);
        neuralNet.setRepresentation(false);

        neuralNet.initializeTrainSet();
        int epoch = 0;
        for (int i = 0; i < 25; i++) {
            neuralNet.initializeWeights();
            epoch += neuralNet.train();
        }
        epoch /= 25;
        System.out.println("average epoch " + epoch);

        // part a
//        String weightFile = "a-weights.txt";
//        String errorFile = "a-trainingErrors.txt";
        // part b
//        String weightFile = "b-weights.txt";
//        String errorFile = "b-trainingErrors.txt";
        // part c
        String weightFile = "c-weights.txt";
        String errorFile = "c-trainingErrors.txt";

        neuralNet.save(new File(weightFile));
        neuralNet.saveError(new File(errorFile));
    }
}

