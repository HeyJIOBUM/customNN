package org.heyjiobum.nn.activation;

public class Linear extends Activation {
    @Override
    public double[] activate(double[] values) {
        return values; // Linear activation is just the identity function
    }

    @Override
    public double[][] jacobian(double[] values) {
        double[][] jacobianMatrix = new double[values.length][values.length];
        for (int i = 0; i < values.length; i++) {
            jacobianMatrix[i][i] = 1;
        }
        return jacobianMatrix;
    }
}