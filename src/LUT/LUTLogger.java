package LUT;

import robocode.RobocodeFileWriter;
import java.io.*;

public class LUTLogger {
    public void writeToFile(File fileToWrite, double winRate, int roundCount) {
        try{
            RobocodeFileWriter fileWriter = new RobocodeFileWriter(fileToWrite.getAbsolutePath(), true);
            fileWriter.write(roundCount + " " + Double.toString(winRate) + "\r\n");
            fileWriter.close();
        }
        catch(Exception e){
            System.out.println(e);
        }
    }

    public void writeLUTValue(File fileToWrite, LUT table) {
        double[][] tableValues = table.getTable();
        int[] visits = table.getVisits();
        try {
            RobocodeFileWriter fileWriter = new RobocodeFileWriter(fileToWrite.getAbsolutePath(), true);
            for (int state = 0; state < tableValues.length; state++){
                for (int action = 0; action < tableValues[state].length; action++){
                    fileWriter.write(Integer.toString(state) + " " +  Integer.toString(action) +
                            " " + Double.toString(tableValues[state][action]) + " " + Integer.toString(visits[state])
                            + "\r\n");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
