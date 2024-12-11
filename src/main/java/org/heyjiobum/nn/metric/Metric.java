package org.heyjiobum.nn.metric;

public abstract class Metric {
    public abstract void clearState();

    public abstract void updateState(double[] yPred, double[] e);

    public abstract String getMetricString();
}