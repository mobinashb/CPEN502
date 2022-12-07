package LUT;

import java.awt.geom.Point2D;

public class Enemy {
    public String name;
    public double bearing;
    public double heading;
    public double changeHeading;
    public double x, y;
    public double distance, speed;
    public long ctime;
    public double energy;

    public Enemy(String name) {
        this.name = name;
        this.energy = 100;
    }

    public Point2D.Double getFuturePos(long gaussTime) {
        double diff = gaussTime - ctime;

        double nextX = x + Math.sin(heading) * speed * diff;
        double nextY = y + Math.cos(heading) * speed * diff;

        Point2D.Double futurePos = new Point2D.Double(nextX, nextY);

        return futurePos;
    }
}
