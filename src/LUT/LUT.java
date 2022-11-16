package LUT;

import Interfaces.LUTInterface;

import java.io.File;
import java.io.IOException;

public class LUT implements LUTInterface {
    private double table[][];

    public LUT() {
        this.table = new double[State.numStates][Action.ROBOT_NUM_ACTIONS];
        initializeLUT();
    }

    @Override
    public void initializeLUT() {
        for(int i=0; i<State.numStates; i++) {
            for(int j=0; j<Action.ROBOT_NUM_ACTIONS; j++) {
                table[i][j] = Math.random();
            }
        }
    }

    public double getQValue(int state, int action) {
        return table[state][action];
    }

    public void setQValue(int state, int action, double value) {
        this.table[state][action] = value;
    }

    public double getMaxValue(int state) {
        double maxValue = -10;
        for(int i=0; i<Action.ROBOT_NUM_ACTIONS; i++) {
            maxValue = Math.max(table[state][i], maxValue);
        }
        return maxValue;
    }

    public int getBestAction(int state) {
        double maxValue = -1000000;
        int action = 0;
        for(int i=0; i<Action.ROBOT_NUM_ACTIONS; i++) {
            if(table[state][i] > maxValue) {
                maxValue = table[state][i];
                action = i;
            }
        }
        return action;
    }

    @Override
    public double outputFor(double[] X) {
        return 0;
    }

    @Override
    public double train(double[] X, double argValue) {
        return 0;
    }

    @Override
    public void save(File argFile) {

    }

    @Override
    public void load(String argFileName) throws IOException {

    }
}
