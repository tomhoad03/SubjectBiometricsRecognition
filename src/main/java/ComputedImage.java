import ai.djl.modality.cv.output.Joints;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.Pixel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ComputedImage {
    private final int id;
    private final MBFImage image;
    private final Boolean isFront;
    private final Pixel centroid;
    private final List<Pixel> boundaryPixels;
    private final Joints joints;
    private DoubleFV extractedFeature;

    public ComputedImage(int id, MBFImage image, Boolean isFront, Pixel centroid, List<Pixel> boundaryPixels, Joints joints) {
        this.id = id;
        this.image = image;
        this.isFront = isFront;
        this.centroid = centroid;
        this.boundaryPixels = boundaryPixels;
        this.joints = joints;
    }

    public int getId() {
        return id;
    }

    public void extractFeature() {
        this.extractedFeature = extractSilhouetteFV().concatenate(extractJointsFV());
    }

    // Extract silhouette feature vector
    public DoubleFV extractSilhouetteFV() {
        int maxBins = 128, count = 0;
        double[] doubleDistances = new double[maxBins];
        ArrayList<PolarPixel> pixels = new ArrayList<>();

        for (Pixel pixel : this.boundaryPixels) {
            double xDiff = pixel.getX() - this.centroid.getX(), yDiff = pixel.getY() - this.centroid.getY();
            double radius = Math.sqrt(Math.pow(xDiff, 2) + Math.pow(yDiff, 2));
            double angle = Math.atan(yDiff / xDiff);

            if (xDiff < 0) {
                angle += Math.PI;
            } else if (xDiff > 0 && yDiff < 0) {
                angle += 2 * Math.PI;
            }
            if (xDiff == 0 && yDiff > 0) {
                angle = Math.PI / 2;
            } else if (xDiff == 0 && yDiff < 0) {
                angle = 3 * (Math.PI / 2);
            } else if (xDiff < 0 && yDiff == 0) {
                angle = Math.PI;
            } else if (xDiff > 0 && yDiff == 0) {
                angle = 0;
            }
            pixels.add(new PolarPixel(radius, angle, pixel));
        }
        pixels.sort(Comparator.comparingDouble(o -> o.angle));

        ArrayList<Double> bin = new ArrayList<>();
        for (PolarPixel pixel : pixels) {
            bin.add(pixel.radius);
            if ((pixel.angle > (((2 * Math.PI) / maxBins) * (count + 1))) || pixel == pixels.get(pixels.size() - 1)) {
                double sum = 0;
                for (double value : bin) {
                    sum += value;
                }
                doubleDistances[count] = sum / bin.size();
                count++;
                bin.clear();
            }
        }
        return new DoubleFV(doubleDistances).normaliseFV();
    }

    public record PolarPixel(double radius, double angle, Pixel pixel) { }

    // Extract joints feature vector
    public DoubleFV extractJointsFV() {
        List<Joints.Joint> joints = this.joints.getJoints();
        ArrayList<Double> jointRadii = new ArrayList<>();
        double width = this.image.getWidth(), height = this.image.getHeight();
        double centroidX = this.centroid.getX() / width, centroidY = this.centroid.getY() / height;

        ArrayList<Pixel> jointPixels = new ArrayList<>();

        for (Joints.Joint joint : joints) {
            Pixel pixel = new Pixel((int) (joint.getX() * width), (int) (joint.getY() * height));
            double radius = Math.sqrt(Math.pow(pixel.getX() - centroidX, 2) + Math.pow(pixel.getY() - centroidY, 2));
            jointRadii.add(radius);
            jointPixels.add(pixel);
        }

        // Invariant features to centroid
        double[] array1 = new double[15];
        double[] array2 = new double[4];
        double[] array3 = new double[5];
        double[] array4 = new double[6];
        for (int i = 0; i < 9; i++) {
            array1[i] = jointRadii.get(i);
        }
        Collections.reverse(jointRadii);
        for (int i = 9; i < 15; i++) {
            array1[i] = jointRadii.get(i - 9);
        }

        // Inter-face distances
        array2[0] = Math.sqrt(Math.pow(jointPixels.get(4).getX() - jointPixels.get(2).getX(), 2) + Math.pow(jointPixels.get(4).getY() - jointPixels.get(2).getY(), 2)); // left ear to left eye
        array2[1] = Math.sqrt(Math.pow(jointPixels.get(2).getX() - jointPixels.get(0).getX(), 2) + Math.pow(jointPixels.get(2).getY() - jointPixels.get(0).getY(), 2)); // left eye to nose
        array2[2] = Math.sqrt(Math.pow(jointPixels.get(0).getX() - jointPixels.get(1).getX(), 2) + Math.pow(jointPixels.get(0).getY() - jointPixels.get(1).getY(), 2)); // nose to right eye
        array2[3] = Math.sqrt(Math.pow(jointPixels.get(1).getX() - jointPixels.get(3).getX(), 2) + Math.pow(jointPixels.get(1).getY() - jointPixels.get(3).getY(), 2)); // right eye to right ear

        // Body widths
        array3[0] = Math.sqrt(Math.pow(jointPixels.get(6).getX() - jointPixels.get(5).getX(), 2) + Math.pow(jointPixels.get(6).getY() - jointPixels.get(5).getY(), 2)); // shoulder to shoulder
        array3[1] = Math.sqrt(Math.pow(jointPixels.get(8).getX() - jointPixels.get(7).getX(), 2) + Math.pow(jointPixels.get(8).getY() - jointPixels.get(7).getY(), 2)); // elbow to elbow
        array3[2] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 5).getX() - jointPixels.get(jointPixels.size() - 6).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 5).getY() - jointPixels.get(jointPixels.size() - 6).getY(), 2)); // hip to hip
        array3[3] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 3).getX() - jointPixels.get(jointPixels.size() - 4).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 3).getY() - jointPixels.get(jointPixels.size() - 4).getY(), 2)); // knee to knee
        array3[4] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 1).getX() - jointPixels.get(jointPixels.size() - 2).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 1).getY() - jointPixels.get(jointPixels.size() - 2).getY(), 2)); // ankle to ankle

        // Body heights
        array4[0] = Math.sqrt(Math.pow(jointPixels.get(6).getX() - jointPixels.get(8).getX(), 2) + Math.pow(jointPixels.get(6).getY() - jointPixels.get(8).getY(), 2)); // left elbow to left shoulder
        array4[1] = Math.sqrt(Math.pow(jointPixels.get(7).getX() - jointPixels.get(5).getX(), 2) + Math.pow(jointPixels.get(7).getY() - jointPixels.get(5).getY(), 2)); // right elbow to right shoulder
        array4[2] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 4).getX() - jointPixels.get(jointPixels.size() - 6).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 4).getY() - jointPixels.get(jointPixels.size() - 6).getY(), 2)); // left hip to left knee
        array4[3] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 3).getX() - jointPixels.get(jointPixels.size() - 5).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 3).getY() - jointPixels.get(jointPixels.size() - 5).getY(), 2)); // right hip to right knee
        array4[4] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 2).getX() - jointPixels.get(jointPixels.size() - 4).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 2).getY() - jointPixels.get(jointPixels.size() - 4).getY(), 2)); // left knee to left ankle
        array4[5] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 1).getX() - jointPixels.get(jointPixels.size() - 3).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 1).getY() - jointPixels.get(jointPixels.size() - 3).getY(), 2)); // right knee to right ankle

        DoubleFV featureVector = new DoubleFV(array1);
        if (isFront) {
            return featureVector.concatenate(new DoubleFV(array2)).concatenate(new DoubleFV(array3)).concatenate(new DoubleFV(array4)).normaliseFV();
        } else {
            return featureVector.concatenate(new DoubleFV(array4)).normaliseFV();
        }
    }

    public DoubleFV getExtractedFeature() {
        return extractedFeature;
    }
}
