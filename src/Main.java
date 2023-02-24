import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Objects;

import static java.awt.image.BufferedImage.TYPE_INT_RGB;

public class Main {
    static final File trainingPath = new File("src/biometrics/training/");
    static final File testingPath = new File("src/biometrics/test/");
    static ArrayList<double[]> trainingHistograms = new ArrayList<>();
    static ArrayList<double[]> testingHistograms = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        // Read the training image pixels
        extractVectors(trainingPath, trainingHistograms);

        // Read the testing image pixels
        extractVectors(testingPath, testingHistograms);

        // Calculate nearest neighbour
        int trainingCount = 0, testingCount = 0, classifiedCount = 0;
        for (double[] testingHistogram : testingHistograms) {
            double minDistance = -1.0;
            int minDistanceHistogram = 0;

            // Calculate distance to all training histograms
            for (double[] trainingHistogram : trainingHistograms) {
                double sum = 0;
                for (int i = 0; i < trainingHistogram.length; i++) {
                    sum += Math.pow((testingHistogram[i] - trainingHistogram[i]), 2);
                }
                double distance = Math.sqrt(sum);

                if (minDistance < 0 || distance < minDistance) {
                    minDistance = distance;
                    minDistanceHistogram = trainingCount;
                }
                trainingCount++;
            }

            // Print the results
            String testingName = Objects.requireNonNull(testingPath.listFiles())[testingCount].getName(), trainingName = Objects.requireNonNull(trainingPath.listFiles())[minDistanceHistogram].getName();
            boolean classificationResult = classificationTest(testingName, trainingName);
            System.out.println("Calculating nearest neighbour of " + testingName + ": " + trainingName + "which gives a classification score " + classificationResult + ".");

            if (classificationResult) {
                classifiedCount++;
            }
            trainingCount = 0;
            testingCount++;
        }

        System.out.println("Classification accuracy: " + ((classifiedCount / testingCount) * 100) + "%");
    }

    private static void extractVectors(File path, ArrayList<double[]> histograms) throws IOException {
        File[] files = path.listFiles();
        ArrayList<int[][]> allPixels = new ArrayList<>();

        for (File file : files) {
            // Get the greyscale pixel values
            int[][] pixels = getGreyscalePixels(file);
            allPixels.add(pixels);

            // Create the histogram
            double[] histogram = createNormalisedHistogram(pixels);
            histograms.add(histogram);

            // Print the results
            System.out.println("Reading image: " + file.getName());
            createGreyscaleImage(pixels, file.getName());
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

    public static void createGreyscaleImage(int[][] pixels, String fileName) throws IOException {
        int width = pixels.length, height = pixels[0].length;
        BufferedImage newImage = new BufferedImage(width, height, TYPE_INT_RGB);

        // Convert the pixel values into an image
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                newImage.setRGB(i, j, new Color(pixels[i][j], pixels[i][j], pixels[i][j]).getRGB());
            }
        }

        // Save the image as a file
        File newFile = new File("src/greyscale/", fileName);
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

    private static boolean classificationTest(String testName, String trainingName) {
        return switch (testName) {
            case "DSC00165.JPG" -> trainingName.equals("021z001ps.jpg");
            case "DSC00166.JPG" -> trainingName.equals("021z001pf.jpg");
            case "DSC00167.JPG" -> trainingName.equals("021z002ps.jpg");
            case "DSC00168.JPG" -> trainingName.equals("021z002pf.jpg");
            case "DSC00169.JPG" -> trainingName.equals("021z003ps.jpg");
            case "DSC00170.JPG" -> trainingName.equals("021z003pf.jpg");
            case "DSC00171.JPG" -> trainingName.equals("021z004ps.jpg");
            case "DSC00172.JPG" -> trainingName.equals("021z004pf.jpg");
            case "DSC00173.JPG" -> trainingName.equals("021z005ps.jpg");
            case "DSC00174.JPG" -> trainingName.equals("021z005pf.jpg");
            case "DSC00175.JPG" -> trainingName.equals("021z006ps.jpg");
            case "DSC00176.JPG" -> trainingName.equals("021z006pf.jpg");
            case "DSC00177.JPG" -> trainingName.equals("021z007ps.jpg");
            case "DSC00178.JPG" -> trainingName.equals("021z007pf.jpg");
            case "DSC00179.JPG" -> trainingName.equals("021z008ps.jpg");
            case "DSC00180.JPG" -> trainingName.equals("021z008pf.jpg");
            case "DSC00181.JPG" -> trainingName.equals("021z009ps.jpg");
            case "DSC00182.JPG" -> trainingName.equals("021z009pf.jpg");
            case "DSC00183.JPG" -> trainingName.equals("021z010ps.jpg");
            case "DSC00184.JPG" -> trainingName.equals("021z010pf.jpg");
            case "DSC00185.JPG" -> trainingName.equals("024z011ps.jpg");
            case "DSC00186.JPG" -> trainingName.equals("024z011pf.jpg");
            default -> false;
        };
    }
}