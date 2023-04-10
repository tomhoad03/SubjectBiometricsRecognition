import ai.djl.modality.cv.output.Joints;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ComputedImage {
    private final int id;
    private final MBFImage image;
    private final ConnectedComponent component;
    private final Joints joints;
    private DoubleFV extractedFeature;

    public ComputedImage(int id, MBFImage image, ConnectedComponent component, Joints joints) {
        this.id = id;
        this.image = image;
        this.component = component;
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
        int maxBins = 90, halfBlankBinSize = 6, count = 0, backCount = 0;
        double[] doubleDistances = new double[maxBins - (halfBlankBinSize * 4)];
        ArrayList<PolarPixel> pixels = new ArrayList<>();
        double centroidX = component.calculateCentroidPixel().getX(), centroidY = component.calculateCentroidPixel().getY();

        for (Pixel pixel : component.getOuterBoundary()) {
            double xDiff = pixel.getX() - centroidX, yDiff = pixel.getY() - centroidY;
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
                if (count < ((maxBins / 4) - halfBlankBinSize)
                        || (count > ((maxBins / 4) + halfBlankBinSize) && count < ((3 * (maxBins / 4)) - halfBlankBinSize))
                        || count > ((3 * (maxBins / 4)) + halfBlankBinSize)) {
                    doubleDistances[count - backCount] = sum / bin.size();
                } else {
                    backCount++;
                }
                count++;
                bin.clear();
            }
        }
        return new DoubleFV(doubleDistances).normaliseFV();
    }

    public record PolarPixel(double radius, double angle, Pixel pixel) { }

    // Extract joints feature vector
    public DoubleFV extractJointsFV() {
        List<Joints.Joint> jointsList = joints.getJoints();
        ArrayList<Double> jointRadii = new ArrayList<>();
        double width = image.getWidth(), height = image.getHeight();
        double centroidX = component.calculateCentroidPixel().getX() / width, centroidY = component.calculateCentroidPixel().getY() / height;

        ArrayList<Pixel> jointPixels = new ArrayList<>();

        for (Joints.Joint joint : jointsList) {
            Pixel pixel = new Pixel((int) (joint.getX() * width), (int) (joint.getY() * height));
            double radius = Math.sqrt(Math.pow(pixel.getX() - centroidX, 2) + Math.pow(pixel.getY() - centroidY, 2));
            jointRadii.add(radius);
            jointPixels.add(pixel);
        }

        // Invariant features to centroid
        double[] array1 = new double[15];
        double[] array2 = new double[4];
        double[] array3 = new double[5];
        double[] array4 = new double[10];
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
        array4[1] = Math.sqrt(Math.pow(jointPixels.get(5).getX() - jointPixels.get(7).getX(), 2) + Math.pow(jointPixels.get(6).getY() - jointPixels.get(7).getY(), 2)); // right elbow to right shoulder
        array4[2] = Math.sqrt(Math.pow(jointPixels.get(6).getX() - jointPixels.get(jointPixels.size() - 6).getX(), 2) + Math.pow(jointPixels.get(6).getY() - jointPixels.get(jointPixels.size() - 6).getY(), 2)); // left hip to left shoulder
        array4[3] = Math.sqrt(Math.pow(jointPixels.get(5).getX() - jointPixels.get(jointPixels.size() - 5).getX(), 2) + Math.pow(jointPixels.get(5).getY() - jointPixels.get(jointPixels.size() - 5).getY(), 2)); // right hip to right shoulder
        array4[4] = Math.sqrt(Math.pow(jointPixels.get(8).getX() - jointPixels.get(jointPixels.size() - 6).getX(), 2) + Math.pow(jointPixels.get(8).getY() - jointPixels.get(jointPixels.size() - 6).getY(), 2)); // left hip to left elbow
        array4[5] = Math.sqrt(Math.pow(jointPixels.get(7).getX() - jointPixels.get(jointPixels.size() - 5).getX(), 2) + Math.pow(jointPixels.get(7).getY() - jointPixels.get(jointPixels.size() - 5).getY(), 2)); // right hip to right elbow
        array4[6] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 4).getX() - jointPixels.get(jointPixels.size() - 6).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 4).getY() - jointPixels.get(jointPixels.size() - 6).getY(), 2)); // left hip to left knee
        array4[7] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 3).getX() - jointPixels.get(jointPixels.size() - 5).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 3).getY() - jointPixels.get(jointPixels.size() - 5).getY(), 2)); // right hip to right knee
        array4[8] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 2).getX() - jointPixels.get(jointPixels.size() - 4).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 2).getY() - jointPixels.get(jointPixels.size() - 4).getY(), 2)); // left knee to left ankle
        array4[9] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 1).getX() - jointPixels.get(jointPixels.size() - 3).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 1).getY() - jointPixels.get(jointPixels.size() - 3).getY(), 2)); // right knee to right ankle

        DoubleFV centroidDistancesFV = new DoubleFV(array1);
        DoubleFV faceDistancesFV = new DoubleFV(array2);
        DoubleFV widthDistancesFV = new DoubleFV(array3);
        DoubleFV heightDistancesFV = new DoubleFV(array4);
        return centroidDistancesFV.concatenate(faceDistancesFV).concatenate(widthDistancesFV).concatenate(heightDistancesFV).normaliseFV();
    }

    public DoubleFV getExtractedFeature() {
        return extractedFeature;
    }
}
