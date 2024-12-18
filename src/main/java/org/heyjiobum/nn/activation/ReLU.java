package org.heyjiobum.nn.activation;

public class ReLU extends Activation {
    @Override
    public double[] activate(double[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = Math.max(0, values[i]);
        }
        return result;
    }

    @Override
    public double[][] jacobian(double[] values) {
        double[][] jacobianMatrix = new double[values.length][values.length];
        for (int i = 0; i < values.length; i++) {
            jacobianMatrix[i][i] = values[i] > 0 ? 1 : 0;
        }
        return jacobianMatrix;
    }
}