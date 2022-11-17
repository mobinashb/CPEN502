package LUT;

import java.awt.Color;
import java.awt.geom.Point2D;
import java.io.File;
import java.util.ArrayList;

import robocode.*;

public class LUTRobot extends AdvancedRobot {
    private static final boolean ON_POLICY = true;
    private static final boolean INTERMEDIATE_REWARD = true;
    private static final boolean BASELINE_ROBOT= false;
    private static final double BASE_DISTANCE = 400.0;

    private Enemy enemy;
    private static LUT table = new LUT();
    private double reward;
    private double firePower = 1;
    private int isHitByBullet = 0;
    private int isHitWall = 0;

    public static final String winningRateFile = LUTRobot.class.getSimpleName() + "-winningRate.log";
    public static final String LUTValueFile = LUTRobot.class.getSimpleName() + "-LUTValues.log";
    static LUTLogger log = new LUTLogger();

    private static int numTotalRounds = 0;
    private static int numWinRounds = 0;

    private static final int ROUNDS_BATCH_SIZE = 100;

    public static final double learningRate = 0.2;
    public static final double discountFactor = 0.9;
    public static double epsilon = 0.9;

    private static final int EPSILON_THRESHOLD = 8000;

    private int prevState = -1;
    private int prevAction = -1;
    private boolean firstRound = true;
    public ArrayList<String> finalStates = new ArrayList<>();

    public void Learn(int currState, int currAction, double reward, boolean isOnPolicy, boolean isIntermidiateRewards) {
        if(!isIntermidiateRewards) {
            finalStates.add(currState+"-"+currAction);
            return;
        }
        double newValue;
        if(firstRound) {
            firstRound = false;
        } else {
            double oldValue = table.getQValue(prevState, prevAction);
            if(isOnPolicy) {
                newValue = oldValue + learningRate * (reward + discountFactor * table.getQValue(currState, currAction)
                        - oldValue);
            } else {
                newValue = oldValue + learningRate * (reward + discountFactor * table.getMaxValue(currState) - oldValue);
            }
            table.setQValue(prevState, prevAction, newValue);
        }
        prevState = currState;
        prevAction = currAction;
    }

    public int getNextAction(int state) {
        if (numTotalRounds > EPSILON_THRESHOLD) epsilon = 0.0;
        double random = Math.random();
        if(random < epsilon) {
            return (int)(Math.random() * Action.ROBOT_NUM_ACTIONS);
        }
        return table.getBestAction(state);
    }

    public void feedReward(double value) {
        int n = finalStates.size();
        double currValue, nextValue;
        String[] strs = finalStates.get(n-1).split("-");
        int state = Integer.valueOf(strs[0]);
        int action = Integer.valueOf(strs[1]);

        table.setQValue(state, action, value);
        nextValue = value;
        for(int i=n-2; i>=0; i--) {
            strs = finalStates.get(i).split("-");
            state = Integer.valueOf(strs[0]);
            action = Integer.valueOf(strs[1]);
            currValue = table.getQValue(state, action);
            currValue += learningRate * (discountFactor * nextValue - currValue);
            table.setQValue(state, action, currValue);
            nextValue = currValue;
        }
    }

    public void run() {
        //state = new State();

        enemy = new Enemy("enemy");
        enemy.distance = 10000;

        setAllColors(Color.red);
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        turnRadarRightRadians(2 * Math.PI);

        while(true) {
            if(!BASELINE_ROBOT) {
                firePower = BASE_DISTANCE / enemy.distance;
                firePower = Math.min(3, firePower);
            }
            radarMovement();
            gunMovement();
            robotMovement();
            execute();
        }
    }

    public void radarMovement() {
        setTurnRadarRightRadians(Double.POSITIVE_INFINITY);
    }

    public void robotMovement() {
        int action;
        if(BASELINE_ROBOT) {
            action = (int)(Math.random() * Action.ROBOT_NUM_ACTIONS);
        } else {
            int state = getState();
            action = getNextAction(state);
            Learn(state, action, reward, ON_POLICY, INTERMEDIATE_REWARD);
            reward = 0.0;
            isHitByBullet = 0;
            isHitWall = 0;
        }

        switch (action) {
            case Action.ROBOT_UP:
                setAhead(Action.ROBOT_MOVE_SHORT_DISTANCE);
                break;
            case Action.ROBOT_UP_LONG:
                setAhead(Action.ROBOT_MOVE_LONG_DISTANCE);
                break;
            case Action.ROBOT_DOWN:
                setBack(Action.ROBOT_MOVE_SHORT_DISTANCE);
                break;
            case Action.ROBOT_DOWN_LONG:
                setBack(Action.ROBOT_MOVE_LONG_DISTANCE);
                break;
            case Action.ROBOT_LEFT:
                setTurnLeft(Action.ROBOT_TURN_DEGREE);
                break;
            case Action.ROBOT_RIGHT:
                setTurnRight(Action.ROBOT_TURN_DEGREE);
                break;
            case Action.ROBOT_FIRE:
                setFire(firePower);
                break;
        }
    }

    private int getState() {
        int heading = State.getHeading(getHeading());
        int bearing = State.getBearing(enemy.bearing);
        int distance = State.getDistance(enemy.distance);
        int energy = State.getEnergyLevel(getEnergy());
        int enemyEnergy = State.getEnergyLevel(enemy.energy);

        int state = State.states[distance][bearing][heading][isHitByBullet][isHitWall][energy][enemyEnergy];

        table.addVisit(state);

        return state;
    }

    private void writeLog(boolean hasWon) {
        numTotalRounds++;
        numWinRounds += hasWon ? 1 : 0;
        if (numTotalRounds % ROUNDS_BATCH_SIZE == 0) {
            double winPercentage = (double) numWinRounds / 100;
            numWinRounds = 0;
            File folderDst = getDataFile(winningRateFile);
            log.writeToFile(folderDst, winPercentage, numTotalRounds / ROUNDS_BATCH_SIZE);
        }
    }

    public void onScannedRobot(ScannedRobotEvent e) {
        if ((enemy.name == e.getName()) || (e.getDistance() < enemy.distance)) {
            enemy.name = e.getName();
            double bearingRadius = (getHeadingRadians() + e.getBearingRadians()) % (2 * Math.PI);
            double heading = normaliseBearing(e.getHeadingRadians() - enemy.heading);
            heading /= (getTime() - enemy.ctime);
            enemy.changeHeading = heading;
            enemy.distance = e.getDistance();
            enemy.x = Math.sin(bearingRadius) * enemy.distance + getX();
            enemy.y = Math.cos(bearingRadius) * enemy.distance + getY();
            enemy.ctime = getTime();
            enemy.speed = e.getVelocity();
            enemy.bearing = e.getBearingRadians();
            enemy.heading = e.getHeadingRadians();
            enemy.energy = e.getEnergy();
        }
    }

    private void gunMovement() {
        long gaussTime, nextTime;
        double gunOffset;
        Point2D.Double p = new Point2D.Double(enemy.x, enemy.y);
        for (int i=0; i<20; i++) {
            nextTime = (int)Math.round((getEuDistance(getX(),getY(),p.x,p.y) / (20 - (3 * firePower))));
            gaussTime = getTime() + nextTime - 10;
            p = enemy.getFuturePos(gaussTime);
        }

        gunOffset = normaliseBearing(getGunHeadingRadians() -
                (Math.PI/2 - Math.atan2(p.y - getY(),p.x -  getX())));
        setTurnGunLeftRadians(gunOffset);
    }

    public double getEuDistance(double x1, double y1, double x2, double y2)
    {
        double x = x1 - x2;
        double y = y1 - y2;
        return Math.sqrt(x * x + y * y);
    }

    double normaliseBearing(double degree) {
        if (degree > Math.PI) {
            degree -= 2*Math.PI;
        }
        if (degree < -Math.PI) {
            degree += 2*Math.PI;
        }
        return degree;
    }

    public void onHitWall(HitWallEvent e){
        isHitWall = 1;
        if(INTERMEDIATE_REWARD) {
            reward -= 3;
        }
    }

    public void onBulletHit(BulletHitEvent e) {
        if(INTERMEDIATE_REWARD) {
            reward += 10;
        }
    }

    public void onHitByBullet(HitByBulletEvent e) {
        isHitByBullet = 1;
        if(INTERMEDIATE_REWARD) {
            reward -= 10;
        }
    }

    public void onBulletMissed(BulletMissedEvent e) {
        if(INTERMEDIATE_REWARD) {
            reward -= 3;
        }
    }

    public void onDeath(DeathEvent event) {
        if(BASELINE_ROBOT) {
            return;
        }
        writeLog(false);
        if (INTERMEDIATE_REWARD) {
            reward -= 60;
        } else {
            feedReward(0);
        }
    }

    public void onWin(WinEvent event) {
        if(BASELINE_ROBOT) {
            return;
        }
        writeLog(true);
        if (INTERMEDIATE_REWARD) {
            reward += 60;
        } else {
            feedReward(1);
        }
    }

    @Override
    public void onBattleEnded(BattleEndedEvent event) {
        File folderDst = getDataFile(LUTValueFile);
        log.writeLUTValue(folderDst, table);
    }
}
