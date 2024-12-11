package org.heyjiobum.nn.activation;

public abstract class Activation {
    public abstract double[] activate(double[] values);
    public abstract double[][] jacobian(double[] values);
}