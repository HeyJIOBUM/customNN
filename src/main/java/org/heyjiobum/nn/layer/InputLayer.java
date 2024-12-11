package org.heyjiobum.nn.layer;

import org.heyjiobum.nn.optimizer.Optimizer;

public class InputLayer extends Layer {
    public InputLayer(int neurons) {
        super(neurons);
    }

    @Override
    public double[][] forward(double[][] inBatch) {
        return inBatch;
    }

    @Override
    public double[][] backward(double[][] deDy, Optimizer optimizer) {
        return deDy;
    }

    @Override
    public void initLayerParams(int prevLayerNeurons) {

    }
}
