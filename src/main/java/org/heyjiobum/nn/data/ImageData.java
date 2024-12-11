package org.heyjiobum.nn.data;

public class ImageData {
    public int imageClass;
    public double[] image;

    public ImageData(double[] image, int imageClass){
        this.image = image;
        this.imageClass = imageClass;
    }

    public double[] getOneHotClass() {
        double[] result = new double[10];
        result[imageClass] = 1;
        return result;
    }
}
