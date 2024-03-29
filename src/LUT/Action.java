package LUT;

public class Action {
    public static final int ROBOT_NUM_ACTIONS = 7;
    public static final double ROBOT_MOVE_SHORT_DISTANCE = 100.0;
    public static final double ROBOT_MOVE_LONG_DISTANCE = 300.0;
    public static final double ROBOT_TURN_DEGREE =  30.0;

    public static final int ROBOT_UP = 0; //moving
    public static final int ROBOT_UP_LONG = 1; //moving
    public static final int ROBOT_DOWN = 2; //moving
    public static final int ROBOT_DOWN_LONG = 3; //moving
    public static final int ROBOT_LEFT = 4; //turning
    public static final int ROBOT_RIGHT = 5; //turning
    public static final int ROBOT_FIRE = 6;

}
