import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import static java.awt.image.BufferedImage.TYPE_INT_RGB;

public class Main {
    public static void main(String[] args) throws IOException {
        File trainingPath = new File("src/biometrics/training/");
        File[] trainingFiles = trainingPath.listFiles();

        // Read the image pixels
        ArrayList<int[][]> trainingPixels = new ArrayList<>();
        ArrayList<double[]> trainingHistograms = new ArrayList<>();
        int idCount = 0;

        for (File trainingFile : trainingFiles) {
            // Get the greyscale pixel values
            int[][] pixels = getGreyscalePixels(trainingFile);
            trainingPixels.add(pixels);

            // Create the histogram
            double[] histogram = createNormalisedHistogram(pixels);
            trainingHistograms.add(histogram);

            // Print the results
            System.out.println("Reading image: " + idCount);
            createGreyscaleImage(pixels, idCount);
            idCount++;
        }
    }

    public static int[][] getGreyscalePixels(File file) throws IOException {
        BufferedImage image = ImageIO.read(file);
        int width = image.getWidth(), height = image.getHeight();
        int[][] pixels = new int[width][height];

        // Convert the pixel values to greyscale
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                Color color = new Color(image.getRGB(i, j), true);
                int colourValue = (color.getRed() + color.getGreen() + color.getBlue()) / 3;
                pixels[i][j] = colourValue;
            }
        }
        return pixels;
    }

    public static void createGreyscaleImage(int[][] pixels, int id) throws IOException {
        int width = pixels.length, height = pixels[0].length;
        BufferedImage newImage = new BufferedImage(width, height, TYPE_INT_RGB);

        // Convert the pixel values into an image
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                newImage.setRGB(i, j, new Color(pixels[i][j], pixels[i][j], pixels[i][j]).getRGB());
            }
        }

        // Save the image as a file
        File newFile = new File("src/greyscale/training/", id + ".jpg");
        newFile.createNewFile();
        ImageIO.write(newImage, "jpg", newFile);
    }

    public static double[] createNormalisedHistogram(int[][] pixels) {
        int totalPixels = pixels.length * pixels[0].length;

        // Create an empty histogram
        double[] histogram = new double[256];
        for (int i = 0; i < 256; i++) {
            histogram[i] = 0;
        }

        // Count the cumulative pixel values
        for (int[] pixelRow : pixels) {
            for (int pixel: pixelRow) {
                histogram[pixel]++;
            }
        }

        // Normalise the histogram
        for (int i = 0; i < histogram.length; i++) {
            histogram[i] = histogram[i] / totalPixels;
        }

        return histogram;
    }
}