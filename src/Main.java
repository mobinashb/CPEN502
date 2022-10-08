import java.io.File;
import java.io.IOException;

public class Main {

    public static void main(String[] args) {
        NeuralNet neuralNet = new NeuralNet();
        neuralNet.setRepresentation(false);
        neuralNet.initializeTrainSet();

        int epoch = 0;
        for (int i = 0; i < 100; i++) {
            neuralNet.initializeWeights();
            epoch += neuralNet.train();
        }
        epoch /= 100;
        System.out.println("average epoch " + epoch);
        neuralNet.saveError();

        neuralNet.save(new File("weights.txt"));
        try {
            neuralNet.load("weights.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

