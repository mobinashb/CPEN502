package LUT;

public class State {

    public enum HP {low, medium, high};
    public enum Distance {close, medium, far};
    public enum DistanceWall {close, medium, far};
    public enum Action {fire, forwardLeft, forwardRight, backwardLeft, backwardRight, forward, backward, left, right};
    public enum Operation {scan, performAction};

}
