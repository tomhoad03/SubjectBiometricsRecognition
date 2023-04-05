import ai.djl.modality.cv.output.Joints;
import org.checkerframework.checker.units.qual.A;
import org.checkerframework.checker.units.qual.C;
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
    private final Pixel centroid;
    private final List<Pixel> boundaryPixels;
    private final double[] firstMoment;
    private final double[] secondCentralisedMoment;
    private final Joints joints;
    private DoubleFV extractedFeature;

    public ComputedImage(int id, MBFImage image, Pixel centroid, List<Pixel> boundaryPixels, double[] firstMoment, double[] secondCentralisedMoment, Joints joints) {
        this.id = id;
        this.image = image;
        this.centroid = centroid;
        this.boundaryPixels = boundaryPixels;
        this.firstMoment = firstMoment;
        this.secondCentralisedMoment = secondCentralisedMoment;
        this.joints = joints;
    }

    public int getId() {
        return id;
    }

    public void extractFeature() {
        this.extractedFeature = extractSilhouetteFV().concatenate(extractJointsFV().concatenate(extractMomentsFV()));
    }

    // Extract silhouette feature vector
    DoubleFV extractSilhouetteFV() {
        int maxBins = 128, count = 0;
        double[] doubleDistances = new double[maxBins];
        ArrayList<PolarPixel> pixels = new ArrayList<>();

        for (Pixel pixel : this.boundaryPixels) {
            if (pixel.getX() == 59 || pixel.getY() == 175) {
                System.out.println("test");
            }

            double xDiff = pixel.getX() - this.centroid.getX(), yDiff = pixel.getY() - this.centroid.getY();
            double radius = Math.sqrt(Math.pow(xDiff, 2) + Math.pow(yDiff, 2));
            double angle = Math.atan(yDiff / xDiff);

            if (xDiff < 0 && (yDiff > 0 || yDiff < 0)) {
                angle += Math.PI;
            } else if (xDiff > 0 && yDiff < 0) {
                angle += (2 * Math.PI);
            }

            pixels.add(new PolarPixel(radius, angle, pixel));
        }
        pixels.sort(Comparator.comparingDouble(o -> o.angle));

        ArrayList<Double> bin = new ArrayList<>();
        for (PolarPixel pixel : pixels) {
            bin.add(pixel.radius);
            if (pixel.angle > (((2 * Math.PI) / maxBins) * (count + 1))) {
                double sum = 0;
                for (double value : bin) {
                    sum += value;
                }
                doubleDistances[count] = sum / bin.size();
                bin.clear();
            }
        }
        return new DoubleFV(doubleDistances).normaliseFV();
    }

    public record PolarPixel(double radius, double angle, Pixel pixel) { }

    // Extract joints feature vector
    DoubleFV extractJointsFV() {
        List<Joints.Joint> joints = this.joints.getJoints();
        ArrayList<Double> jointRadii = new ArrayList<>();
        double width = this.image.getWidth(), height = this.image.getHeight();
        double centroidX = this.centroid.getX() / width, centroidY = this.centroid.getY() / height;

        for (Joints.Joint joint : joints) {
            Pixel pixel = new Pixel((int) (joint.getX() * width), (int) (joint.getY() * height));
            double radius = Math.sqrt(Math.pow(pixel.getX() - centroidX, 2) + Math.pow(pixel.getY() - centroidY, 2));
            jointRadii.add(radius);
        }

        // Face, shoulders, elbows, hips, legs and feet joints
        double[] array1 = new double[17];
        for (int i = 0; i < 9; i++) {
            array1[i] = jointRadii.get(i);
        }
        Collections.reverse(jointRadii);
        for (int i = 9; i < 15; i++) {
            array1[i] = jointRadii.get(i - 9);
        }
        return new DoubleFV(array1).normaliseFV();
    }

    // Extract moments feature vector
    DoubleFV extractMomentsFV() {
        return new DoubleFV(this.secondCentralisedMoment).normaliseFV();
    }

    public DoubleFV getExtractedFeature() {
        return extractedFeature;
    }
}
