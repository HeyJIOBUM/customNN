package org.heyjiobum.nn.activation;

public class Softmax extends Activation {
    @Override
    public double[] activate(double[] values) {
        double[] expValues = new double[values.length];
        double sumExp = 0;

        for (int i = 0; i < values.length; i++) {
            expValues[i] = Math.exp(values[i]);
            sumExp += expValues[i];
        }

        for (int i = 0; i < expValues.length; i++) {
            expValues[i] /= sumExp; // Normalize to get probabilities
        }

        return expValues;
    }

    @Override
    public double[][] jacobian(double[] values) {
        double[] softmaxActivations = activate(values);
        int n = values.length;
        double[][] jacobianMatrix = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    jacobianMatrix[i][j] = softmaxActivations[i] * (1 - softmaxActivations[i]);
                } else {
                    jacobianMatrix[i][j] = -softmaxActivations[i] * softmaxActivations[j];
                }
            }
        }
        return jacobianMatrix;
    }
}