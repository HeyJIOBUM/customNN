package org.heyjiobum.nn.metric;

public class MseLossMetric extends Metric {
    private int iterations;
    private double overallLoss;

    @Override
    public void clearState() {
        iterations = 0;
        overallLoss = 0;
    }

    @Override
    public void updateState(double[] yPred, double[] e) {
        iterations += 1;

        for (int i = 0; i < yPred.length; i++) {
            overallLoss += Math.pow(yPred[i] - e[i], 2) / 2;
        }
    }

    @Override
    public String getMetricString() {
        return "mse_loss: " + overallLoss / iterations;
    }
}
