package LUT;

import robocode.RobocodeFileWriter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Statistics {
    private static final int PERIOD = 100;
    private static ArrayList<Integer> scoreList = new ArrayList<>();
    private int numRounds;

    public Statistics(int numRounds) {
        this.numRounds = numRounds;
    }

    public static void saveScore(int win) {
        scoreList.add(win);
    }

    public static void printScoreTable() {
        int size = scoreList.size();
        for(int i=0; i<size; i++) {
            System.out.println("score list: " + i + " " + scoreList.get(i));
        }
    }

    public static void printWinRates(LogFile logFile, File fileToWrite) {
        int size = scoreList.size();
        int groups = size / PERIOD;
        double cnt;
        FileWriter resultWriter = null;

        for(int i=0; i<groups; i++) {
            cnt = 0.0;
            for(int j=0; j<PERIOD; j++) {
                cnt += scoreList.get(i*PERIOD + j);
            }
            logFile.writeToFile(fileToWrite, (cnt/PERIOD), i+1);
        }


    }
}
