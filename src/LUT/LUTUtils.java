package LUT;

public class LUTUtils {

    public static double[] oneHotEncoding(double val, int encodingLength){
        double[] result = new double[encodingLength];
        for (int i = 0; i < encodingLength; i++){
            result[i] = 0;
        }
        result[(encodingLength-1)-(int)val] = 1;
        return result;
    }

}
