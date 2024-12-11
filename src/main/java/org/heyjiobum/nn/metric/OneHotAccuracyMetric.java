package org.heyjiobum.nn.metric;

public class OneHotAccuracyMetric extends Metric {
    private int iterations;
    private double guessed_counter;

    @Override
    public void clearState() {
        iterations = 0;
        guessed_counter = 0;
    }

    @Override
    public void updateState(double[] yPred, double[] e) {
        iterations += 1;

        int yPredMaxIndex = 0;
        for (int i = 1; i < yPred.length; i++) {
            if (yPred[i] > yPred[yPredMaxIndex])
                yPredMaxIndex = i;
        }

        int eMaxIndex = 0;
        for (int i = 1; i < e.length; i++) {
            if (e[i] > e[eMaxIndex])
                eMaxIndex = i;
        }

        if (yPredMaxIndex == eMaxIndex)
            guessed_counter += 1;
    }

    @Override
    public String getMetricString() {
        return "accuracy: " + 100 * iterations / guessed_counter;
    }
}
