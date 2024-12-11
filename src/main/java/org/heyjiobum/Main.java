package org.heyjiobum;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import org.heyjiobum.nn.Model;
import org.heyjiobum.nn.activation.Linear;
import org.heyjiobum.nn.activation.ReLU;
import org.heyjiobum.nn.activation.Softmax;
import org.heyjiobum.nn.data.ImageData;
import org.heyjiobum.nn.layer.InputLayer;
import org.heyjiobum.nn.layer.Layer;
import org.heyjiobum.nn.layer.LinearLayer;
import org.heyjiobum.nn.metric.Metric;
import org.heyjiobum.nn.metric.MseLossMetric;
import org.heyjiobum.nn.metric.OneHotAccuracyMetric;
import org.heyjiobum.nn.optimizer.Adam;
import org.heyjiobum.nn.optimizer.SGD;

import java.awt.*;
import java.awt.image.ColorModel;
import java.io.File;
import java.util.*;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Main {
    final private static String PATH = "D:\\JetBrains\\Projects\\IntelIjProjects\\customNN\\dataset\\mnist_png\\";

    public static void main(String[] args) {

        ImageData[] train = getImages(PATH + "train", true);
        ImageData[] test = getImages(PATH + "test", false);

        ImageData[][] trainBatches = divideInBatches(train, 100);
        ImageData[][] testBatches = divideInBatches(test, 100);

        Model model = new Model
                (
                    new Layer[]{
                            new InputLayer(28 * 28),
                            new LinearLayer(128, new ReLU()),
                            new LinearLayer(10, new Linear())
                    },
//                    new Adam(0.001, 0.9, 0.999, 1e-10),
                    new SGD(0.001),
                    new Metric[]{
                            new OneHotAccuracyMetric(),
                            new MseLossMetric()
                    }
                );

        model.fit(trainBatches, testBatches, 1000);
    }

    private static ImageData[][] divideInBatches (ImageData[] trainData, int batchSize){
        List<ImageData> dataList = Arrays.asList(trainData);
        Collections.shuffle(dataList);
        trainData = dataList.toArray(new ImageData[0]);

        int numBatches = (int) Math.ceil((double) trainData.length / batchSize);
        ImageData[][] dividedData = new ImageData[numBatches][];

        for (int i = 0; i < numBatches; i++) {
            int start = i * batchSize;
            int end = Math.min(start + batchSize, trainData.length);
            dividedData[i] = Arrays.copyOfRange(trainData, start, end);
        }

        return dividedData;
    }

    public static ImageData[] getImages (String path,boolean isProcess){
        List<ImageData> imagesList = new ArrayList<>();
        List<Future<List<ImageData>>> futures = new ArrayList<>();

        File directory = new File(path);
        if (directory.isDirectory()) {
            File[] files = directory.listFiles();
            if (files != null) {
                ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

                for (File file : files) {
                    Future<List<ImageData>> future = executor.submit(() ->
                            getImagesForClass(file.getAbsolutePath(), Integer.parseInt(file.getName()), isProcess)
                    );
                    futures.add(future);
                }

                for (Future<List<ImageData>> future : futures) {
                    try {
                        imagesList.addAll(future.get());
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    } catch (ExecutionException e) {
                        throw new RuntimeException(e);
                    }
                }

                executor.shutdown();
            }
        } else {
            System.out.println("Указанный путь не является каталогом.");
        }

        return imagesList.toArray(new ImageData[0]);
    }

    public static List<ImageData> getImagesForClass (String path,int imageClass, boolean isProcess){
        List<ImageData> oneClassImages = new ArrayList<>();

        File directory = new File(path);
        File[] files = directory.listFiles();

        for (File file : files) {
            System.out.println(file.getAbsolutePath());
            ImagePlus imp = IJ.openImage(file.getAbsolutePath());
            ImageProcessor ip = imp.getProcessor();

            if (isProcess) {
                Random rnd = new Random();
                ip.setBackgroundColor(new Color(0));
                ip.rotate(rnd.nextDouble(360));
                ip.translate(rnd.nextDouble(-7, 7),
                        rnd.nextDouble(-7, 7));
            }

            double[] image = imageToArray(ip);
            oneClassImages.add(new ImageData(image, imageClass));
        }

        return oneClassImages;
    }

    public static double[] imageToArray (ImageProcessor ip){
        ColorModel colorModel = ip.getColorModel();
        byte[] bytePixels = (byte[]) ip.getPixels();
        double[] binaryPixels = new double[bytePixels.length];
        int threshold = 128;

        for (int i = 0; i < bytePixels.length; ++i) {
            int index = bytePixels[i] & 0xFF;
            int rgb = colorModel.getRGB(index);
            Color color = new Color(rgb);

            int red = color.getRed();
            int green = color.getGreen();
            int blue = color.getBlue();

            int brightness = (int) (0.299 * red + 0.587 * green + 0.114 * blue);

            binaryPixels[i] = (brightness > threshold) ? 1 : 0;
        }

        return binaryPixels;
    }
}
