import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;

import static java.awt.image.BufferedImage.TYPE_BYTE_GRAY;
import static java.awt.image.BufferedImage.TYPE_INT_RGB;

public class Main {
    public static void main(String[] args) throws IOException {
        File trainingPath = new File("src/biometrics/training/");
        File[] trainingFiles = trainingPath.listFiles();

        // Read the image pixels
        BufferedImage firstImage = ImageIO.read(trainingFiles[0]);
        int width = firstImage.getWidth(), height = firstImage.getHeight();

        ArrayList<int[][]> trainingPixels = new ArrayList<>();
        int[][][] allTrainingPixels = new int[width][height][trainingFiles.length];
        int countA = 0;

        for (File trainingFile : trainingFiles) {
            BufferedImage image = ImageIO.read(trainingFile);
            int[][] pixels = new int[width][height];

            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    int pixelValue = image.getRGB(i, j);

                    Color color = new Color(pixelValue, true);
                    int colourValue = (color.getRed() + color.getGreen() + color.getBlue()) / 3;

                    pixels[i][j] = colourValue;
                    allTrainingPixels[i][j][countA] = colourValue;
                }
            }

            trainingPixels.add(pixels);

            System.out.println("Reading image: " + countA);
            countA++;
        }

        // Calculate the background
        int[][] backgroundPixels = new int[width][height];

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                ArrayList<Integer> pixels = new ArrayList<>();

                for (int k = 0; k < trainingFiles.length; k++) {
                    pixels.add(allTrainingPixels[i][j][k]);
                }

                Collections.sort(pixels);

                if (pixels.size() % 2 == 0) {
                    backgroundPixels[i][j] = pixels.get(pixels.size() / 2) + pixels.get((pixels.size() / 2) - 1);
                } else {
                    backgroundPixels[i][j] = pixels.get((pixels.size() - 1) / 2) - 1;
                }
            }
        }

        // Remove the backgrounds
        Files.createDirectories(Paths.get("src/nobackground/training/"));
        int countB = 0;

        for (int[][] pixels : trainingPixels) {
            BufferedImage newImage = new BufferedImage(width, height, TYPE_INT_RGB);

            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    if ((pixels[i][j] < backgroundPixels[i][j] + 50) && (pixels[i][j] > backgroundPixels[i][j] - 50)) {
                        newImage.setRGB(i, j, Color.white.getRGB());
                    } else {
                        newImage.setRGB(i, j, new Color(pixels[i][j], pixels[i][j], pixels[i][j]).getRGB());
                    }
                }
            }

            File newFile = new File("src/nobackground/training/", countB + ".jpg");
            newFile.createNewFile();
            ImageIO.write(newImage, "jpg", newFile);

            System.out.println("Removing background from image: " + countB);
            countB++;
        }
    }
}