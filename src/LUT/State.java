package LUT;

public class State {
    public static int numStates;

    public static final int numEnemyDistance = 10; //number of enemy distances
    public static final int numEnemyDirection = 4; //number of enemy directions
    public static final int numRobotDirection = 4; //number of robot directions
    public static final int numHitByBullet = 2; //whether hit by bullet or not
    public static final int numHitWall = 2; //whether hit wall or not
    public static final int numEnergy = 5; //levels of robot energy

    public static final double totalAngle = 360.0;
    public static final double circle = Math.PI * 2;

    public static int states[][][][][][];

    static {
        states = new int[numEnemyDistance][numEnemyDirection][numRobotDirection][numHitByBullet][numHitWall][numEnergy];
        int totalStates = 0;
        for(int a = 0; a < numEnemyDistance; a++) {
            for(int b = 0; b < numEnemyDirection; b++) {
                for(int c = 0; c < numRobotDirection; c++) {
                    for(int d = 0; d < numHitByBullet; d++) {
                        for(int e = 0; e < numHitByBullet; e++) {
                            for(int f = 0; f < numEnergy; f++) {
                                states[a][b][c][d][e][f] = totalStates++;
                            }
                        }
                    }
                }
            }
        }
        numStates = totalStates;
    }

    public static int getHeading(double heading) {
        double angle = totalAngle / numRobotDirection;
        double newHeading = heading+angle/2;
        while (newHeading > totalAngle) {
            newHeading-=totalAngle;
        }
        return (int)(newHeading / angle);
    }

    public static int getBearing(double bearing) {
        double angle= circle / numEnemyDirection;
        double newBearing = bearing;
        if(bearing < 0) {
            newBearing += circle;
        }
        newBearing += angle / 2;
        if(newBearing > circle) {
            newBearing = newBearing - circle;
        }
        return (int) (newBearing / angle);
    }

    public static int getDistance(double distance) {
        int res = (int)(distance / 100.0);
        return Math.min(numEnemyDistance-1, res);
    }

    public static int getEnergyLevel(double energy) {
        double levels = 100 / numEnergy;
        return Math.min((int)(energy/levels), numEnergy-1);
    }
}
